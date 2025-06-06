import requests
import base64
import io
import numpy as np
import torch
from PIL import Image
import json
import time # Import time for polling delay

# --- Configuration ---
# Updated API endpoint for FLUX Kontext Max
API_ENDPOINT = "https://api.bfl.ai/v1/flux-kontext-max" 
# Polling endpoint (usually a base URL for results) - though we'll use the one from polling_url
POLLING_BASE_URL = "https://api.us1.bfl.ai/v1/get_result" 

# Common aspect ratios for user convenience
ASPECT_RATIOS_T2I = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "3:2", "2:3"]
# For editing, None means API might infer from image or use a default
ASPECT_RATIOS_EDIT = ["None", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "3:2", "2:3"]

# Polling parameters
POLLING_INTERVAL_SECONDS = 3 # How often to check the status
POLLING_TIMEOUT_SECONDS = 300 # Max time to wait for results (5 minutes)


# --- Helper Functions ---

def tensor_to_base64(single_image_tensor: torch.Tensor) -> str:
    """
    Converts a single ComfyUI image tensor (H, W, C) to a base64 encoded string.
    Input tensor is expected to have values in the range [0.0, 1.0].
    """
    if not isinstance(single_image_tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(single_image_tensor)}")
    
    if single_image_tensor.ndim != 3: # Should be H, W, C
        raise ValueError(f"Expected a 3D tensor (H, W, C) for single image conversion, but got {single_image_tensor.ndim}D with shape {single_image_tensor.shape}")
    
    numpy_image = single_image_tensor.cpu().numpy()
    # Ensure it's in 0-1 range before multiplying by 255
    numpy_image = np.clip(numpy_image, 0.0, 1.0)
    
    image_pil = Image.fromarray((numpy_image * 255).astype(np.uint8), 'RGB')
    
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG") # PNG is lossless
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str

def base64_to_tensor(base64_str: str) -> torch.Tensor:
    """
    Converts a base64 encoded image string to a ComfyUI image tensor (1, H, W, C).
    Output tensor will have values in the range [0.0, 1.0].
    """
    try:
        image_data = base64.b64decode(base64_str)
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding base64 string or opening image: {e}")
        # Return a small black tensor as a fallback
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

    numpy_image = np.array(image_pil, dtype=np.float32) / 255.0
    torch_tensor = torch.from_numpy(numpy_image) # Shape: (H, W, C)
    
    if torch_tensor.ndim == 3: # H, W, C
        torch_tensor = torch_tensor.unsqueeze(0) # Add batch dimension -> (1, H, W, C)
    return torch_tensor

def create_empty_image_tensor(height=64, width=64):
    """Creates a small black image tensor (1, H, W, C) for error cases."""
    return torch.zeros((1, height, width, 3), dtype=torch.float32)

def process_image_response(img_response: requests.Response) -> torch.Tensor:
    """
    Processes an image HTTP response into a ComfyUI tensor.
    Similar to process_image_response from comfy_api_nodes.apinode_utils but
    adapted for direct use.
    """
    img_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    image_pil = Image.open(io.BytesIO(img_response.content)).convert("RGB")
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)[None,] # Add batch dimension
    return image_tensor


def poll_for_results(polling_url, api_key, timeout=POLLING_TIMEOUT_SECONDS, interval=POLLING_INTERVAL_SECONDS):
    """
    Polls the given URL for results until status is 'succeeded', 'Ready', 'failed', or timeout.
    """
    start_time = time.time()
    headers = {
        "x-key": api_key,
        "Content-Type": "application/json"
    }

    while time.time() - start_time < timeout:
        try:
            poll_response = requests.get(polling_url, headers=headers, timeout=60) # Timeout for polling request
            poll_response.raise_for_status()
            poll_result = poll_response.json()

            status = poll_result.get("status", "").lower() # Convert to lowercase immediately
            
            # If "Ready" is a success status, include it here
            if status == "succeeded" or status == "ready": # Now compares lowercase status
                print(f"Kontext API: Polling succeeded or Ready, results retrieved. Status: {status}")
                return poll_result
            elif status == "failed":
                error_message = poll_result.get("message", "Task failed")
                detail = poll_result.get("detail", "")
                print(f"\033[91mKontext API Polling Error: Task failed - {error_message} {detail}\033[0m")
                print(f"Full Polling Response: {json.dumps(poll_result, indent=2)}")
                return None # Indicate failure
            elif status in ["pending", "processing"]: # Now compares lowercase status
                print(f"Kontext API: Task status: {status}. Waiting for results...")
                time.sleep(interval)
            else:
                print(f"\033[93mKontext API Polling Warning: Unexpected status '{status}'. Retrying...\033[0m")
                print(f"Full API Response for unexpected status '{status}': {json.dumps(poll_result, indent=2)}")
                time.sleep(interval)

        except requests.exceptions.HTTPError as http_err:
            print(f"\033[91mHTTP error during polling: {http_err}\033[0m")
            if http_err.response is not None:
                try:
                    print(f"Response content: {http_err.response.json()}")
                except json.JSONDecodeError:
                    print(f"Response content: {http_err.response.text}")
            return None
        except requests.exceptions.RequestException as req_err:
            print(f"\033[91mRequest error during polling: {req_err}\033[0m")
            return None
        except Exception as e:
            print(f"\033[91mAn unexpected error occurred during polling: {e}\033[0m")
            import traceback
            traceback.print_exc()
            return None
    
    print(f"\033[91mKontext API Polling Timeout: No results after {timeout} seconds.\033[0m")
    return None # Indicate timeout


# --- ComfyUI Nodes ---

class KontextTextToImageOfficialAPI:
    """
    Generates images from text prompts using the official FLUX Kontext Max API (Text-to-Image).
    Now includes asynchronous polling for results, and expanded control parameters.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Your Kontext API Key (e.g., e9e91dfe-...)"}),
                "prompt": ("STRING", {"default": "A beautiful landscape painting", "multiline": True}),
                "aspect_ratio": (ASPECT_RATIOS_T2I, {"default": "1:1"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 150, "step": 1, "tooltip": "Number of inference steps."}), # Added steps
                "prompt_upsampling": ("BOOLEAN", {"default": False, "tooltip": "Enable creative prompt upsampling (nondeterministic)."}) # Added prompt_upsampling
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}), # API uses its own random if not provided or -1
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_text_to_image"
    CATEGORY = "Kontext Official API (Max)"

    def generate_text_to_image(self, api_key, prompt, aspect_ratio, num_images, 
                               guidance_scale, steps, prompt_upsampling, seed=None): # Added steps and prompt_upsampling
        if not api_key:
            print("\033[91mError: Kontext API Key is missing.\033[0m")
            return (create_empty_image_tensor(),)

        headers = {
            "x-key": api_key, # Updated header key
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps, # Added steps to payload
            "prompt_upsampling": prompt_upsampling, # Added prompt_upsampling to payload
            "return_type": "base64" # We will process base64
        }

        if seed is not None and seed != -1: # API doc implies seed is optional, sending if user provides a valid one
            payload["seed"] = seed

        print(f"Kontext Text-to-Image (Max): Sending initial request with prompt '{prompt[:50]}...' to {API_ENDPOINT}")

        try:
            response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=120) 
            response.raise_for_status() 
            
            initial_result = response.json()
            
            # Print initial result for debugging
            print(f"Full Initial API Response: {json.dumps(initial_result, indent=2)}")

            output_images = [] # This will now store torch.Tensor objects

            # Check if it's an immediate success or polling is needed
            if "polling_url" in initial_result:
                polling_url = initial_result["polling_url"]
                print(f"Kontext Text-to-Image (Max): Task submitted. Polling for results at: {polling_url}")
                final_result = poll_for_results(polling_url, api_key)

                # Print final result from polling for debugging
                print(f"Full Final API Result after polling: {json.dumps(final_result, indent=2)}")

                if final_result and (final_result.get("status", "").lower() == "succeeded" or final_result.get("status", "").lower() == "ready"):
                    # Extract image URL from 'result.sample'
                    if "result" in final_result and "sample" in final_result["result"]:
                        img_url = final_result["result"]["sample"]
                        print(f"Kontext Text-to-Image (Max): Image URL found: {img_url}")
                        # Fetch the image from the URL
                        img_response = requests.get(img_url, timeout=60)
                        tensor_img = process_image_response(img_response)
                        output_images.append(tensor_img)
                    else:
                        print("\033[91mKontext API Error (Text-to-Image Max): 'result' or 'sample' field missing in successful polling response.\033[0m")
                        return (create_empty_image_tensor(),)
                else:
                    print("\033[91mKontext API Error (Text-to-Image Max): Polling did not return successful results or timed out.\033[0m")
                    return (create_empty_image_tensor(),)
            # This 'else' block handles synchronous success (if API ever sends base64 directly in initial response)
            elif (initial_result.get("status", "").lower() == "succeeded" or initial_result.get("status", "").lower() == "ready") and "data" in initial_result and "images" in initial_result["data"]:
                print(f"Kontext Text-to-Image (Max): Immediate success or Ready status detected. Status: {initial_result.get('status')}")
                # This branch handles the case where the API *might* return base64 images directly, though sample URL is more likely
                for b64_img_str in initial_result["data"]["images"]:
                    output_images.append(base64_to_tensor(b64_img_str))
            else:
                error_message = initial_result.get("message", "Unknown API error in initial response")
                detail = initial_result.get("detail", "")
                print(f"\033[91mKontext API Error (Text-to-Image Max): {error_message} {detail}\033[0m")
                return (create_empty_image_tensor(),)

            if not output_images:
                print("\033[93mWarning: API succeeded but returned no images from URL or base64 data.\033[0m")
                return (create_empty_image_tensor(),)
            
            print(f"Kontext Text-to-Image (Max): Successfully generated {len(output_images)} image(s).")
            # If we collected a list of tensors directly, concatenate them
            return (torch.cat(output_images, dim=0),)

        except requests.exceptions.HTTPError as http_err:
            print(f"\033[91mHTTP error occurred (Text-to-Image Max): {http_err}\033[0m")
            if http_err.response is not None:
                try:
                    print(f"Response content: {http_err.response.json()}")
                except json.JSONDecodeError:
                    print(f"Response content: {http_err.response.text}")
            return (create_empty_image_tensor(),)
        except requests.exceptions.RequestException as req_err: # This catches NameResolutionError etc.
            print(f"\033[91mRequest error occurred (Text-to-Image Max): {req_err}\033[0m")
            return (create_empty_image_tensor(),)
        except Exception as e:
            print(f"\033[91mAn unexpected error occurred (Text-to-Image Max): {e}\033[0m")
            import traceback
            traceback.print_exc()
            return (create_empty_image_tensor(),)


class KontextImageEditingOfficialAPI:
    """
    Edits images based on text prompts using the official FLUX Kontext Max API (Image Editing).
    Now includes asynchronous polling for results, and expanded control parameters.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Your Kontext API Key (e.g., e9e91dfe-...)"}),
                "prompt": ("STRING", {"default": "Make this image look like a watercolor painting", "multiline": True}),
                "image": ("IMAGE",), 
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}), # Added strength
                "steps": ("INT", {"default": 50, "min": 1, "max": 150, "step": 1, "tooltip": "Number of inference steps."}), # Added steps
                "prompt_upsampling": ("BOOLEAN", {"default": False, "tooltip": "Enable creative prompt upsampling (nondeterministic)."}) # Added prompt_upsampling
            },
            "optional": {
                "aspect_ratio": (ASPECT_RATIOS_EDIT, {"default": "None"}), 
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image_editing"
    CATEGORY = "Kontext Official API (Max)"

    def generate_image_editing(self, api_key, prompt, image: torch.Tensor, num_images, 
                                 guidance_scale, strength, steps, prompt_upsampling, aspect_ratio="None", seed=None): # Added steps and prompt_upsampling
        if not api_key:
            print("\033[91mError: Kontext API Key is missing.\033[0m")
            return (create_empty_image_tensor(),)

        image_to_process = None
        if image.ndim == 4: 
            if image.shape[0] > 1:
                print(f"\033[93mWarning: Image editing node received a batch of {image.shape[0]} images. Processing only the first image.\033[0m")
            image_to_process = image[0] 
        elif image.ndim == 3: 
             image_to_process = image
        else:
            print(f"\033[91mError: Unexpected image tensor dimensions: {image.ndim}. Expected 3 or 4.\033[0m")
            return (create_empty_image_tensor(),)

        try:
            base64_input_image = tensor_to_base64(image_to_process)
        except Exception as e:
            print(f"\033[91mError converting input image to base64: {e}\033[0m")
            import traceback
            traceback.print_exc()
            return (create_empty_image_tensor(),)

        headers = {
            "x-key": api_key, # Updated header key
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": prompt,
            "input_image": base64_input_image, # Corrected to 'input_image'
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "strength": strength, 
            "steps": steps, # Added steps to payload
            "prompt_upsampling": prompt_upsampling, # Added prompt_upsampling to payload
            "return_type": "base64"
        }
        print(f"Kontext Image Editing (Max): Sending payload with 'input_image' key and other parameters.") # Updated print statement

        if aspect_ratio != "None" and aspect_ratio is not None: # API doc says aspect_ratio is optional for editing
            payload["aspect_ratio"] = aspect_ratio
        
        if seed is not None and seed != -1:
            payload["seed"] = seed
        
        print(f"Kontext Image Editing (Max): Sending initial request with prompt '{prompt[:50]}...' to {API_ENDPOINT}")

        try:
            response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=180) 
            response.raise_for_status()
            
            initial_result = response.json()
            
            # Print initial result for debugging
            print(f"Full Initial API Response: {json.dumps(initial_result, indent=2)}")

            output_images = [] # Initialize list for images

            # Check if it's an immediate success or polling is needed
            if "polling_url" in initial_result:
                polling_url = initial_result["polling_url"]
                print(f"Kontext Image Editing (Max): Task submitted. Polling for results at: {polling_url}")
                final_result = poll_for_results(polling_url, api_key)

                # Print final result from polling for debugging
                print(f"Full Final API Result after polling: {json.dumps(final_result, indent=2)}")

                if final_result and (final_result.get("status", "").lower() == "succeeded" or final_result.get("status", "").lower() == "ready"):
                    # Extract image URL from 'result.sample'
                    if "result" in final_result and "sample" in final_result["result"]:
                        img_url = final_result["result"]["sample"]
                        print(f"Kontext Image Editing (Max): Image URL found: {img_url}")
                        # Fetch the image from the URL
                        img_response = requests.get(img_url, timeout=60)
                        tensor_img = process_image_response(img_response)
                        output_images.append(tensor_img)
                    else:
                        print("\033[91mKontext API Error (Image Editing Max): 'result' or 'sample' field missing in successful polling response.\033[0m")
                        return (create_empty_image_tensor(),)
                else:
                    print("\033[91mKontext API Error (Image Editing Max): Polling did not return successful results or timed out.\033[0m")
                    return (create_empty_image_tensor(),)
            # This 'else' block handles synchronous success (if API ever sends base64 directly in initial response)
            elif (initial_result.get("status", "").lower() == "succeeded" or initial_result.get("status", "").lower() == "ready") and "data" in initial_result and "images" in initial_result["data"]:
                print(f"Kontext Image Editing (Max): Immediate success or Ready status detected. Status: {initial_result.get('status')}")
                # This branch handles the case where the API *might* return base64 images directly, though sample URL is more likely
                for b64_img_str in initial_result["data"]["images"]:
                    output_images.append(base64_to_tensor(b64_img_str))
            else:
                error_message = initial_result.get("message", "Unknown API error in initial response")
                detail = initial_result.get("detail", "")
                print(f"\033[91mKontext API Error (Image Editing Max): {error_message} {detail}\033[0m")
                return (create_empty_image_tensor(),)

            if not output_images:
                print("\033[93mWarning: API succeeded but returned no images from URL or base64 data.\033[0m")
                return (create_empty_image_tensor(),)
            
            print(f"Kontext Image Editing (Max): Successfully generated {len(output_images)} image(s)。")
            return (torch.cat(output_images, dim=0),)

        except requests.exceptions.HTTPError as http_err:
            print(f"\033[91mHTTP error occurred (Image Editing Max): {http_err}\033[0m")
            if http_err.response is not None:
                try:
                    print(f"Response content: {http_err.response.json()}")
                except json.JSONDecodeError:
                    print(f"Response content: {http_err.response.text}")
            return (create_empty_image_tensor(),)
        except requests.exceptions.RequestException as req_err: # This catches NameResolutionError etc.
            print(f"\033[91mRequest error occurred (Image Editing Max): {req_err}\033[0m")
            return (create_empty_image_tensor(),)
        except Exception as e:
            print(f"\033[91mAn unexpected error occurred (Image Editing Max): {e}\033[0m")
            import traceback
            traceback.print_exc()
            return (create_empty_image_tensor(),)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "KontextTextToImageOfficialAPI_Max": KontextTextToImageOfficialAPI, # Renamed to avoid conflicts if old one exists
    "KontextImageEditingOfficialAPI_Max": KontextImageEditingOfficialAPI, # Renamed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KontextTextToImageOfficialAPI_Max": "Kontext Text-to-Image (Official Max)",
    "KontextImageEditingOfficialAPI_Max": "Kontext Image Editing (Official Max)",
}
