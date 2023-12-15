import json


def map_model_names(model_path):
    model_name_mapping = {
        "ljspeech_parallel_wavegan": "PWG",
        "ljspeech_hifiGAN": "HiFi-GAN",
        "ljspeech_melgan": "MelGAN",
        "ljspeech_full_band_melgan": "FB-MelGAN",
        "ljspeech_multi_band_melgan": "MB-MelGAN",
        "ljspeech_waveglow": "WaveGlow",
    }

    for model_name, short_name in model_name_mapping.items():
        if model_name in model_path:
            return short_name

    # If no match is found, return the original model name
    return model_path
 


def print_eer_values(json_data):
    if isinstance(json_data, list):
        # If it's a list, assume it's an array of objects
        for item in json_data:
            if isinstance(item, dict):
                # If the element is a dictionary, print the value of "eer" for each column
                for key, value in item.items():
                    if key == "eer":
                        print(f"Column: {key}, Value: {value}")
            else:
                print("Invalid item in the list")
    elif isinstance(json_data, dict):
        # If it's a dictionary, assume it's a single object
        for key, value in json_data.items():
            print(f"{map_model_names(key)}, {json_data[key]['eer']}")
            for model in json_data[key]['out_distribution'].keys():
                eer_value = json_data[key]['out_distribution'][model]['eer']
                eer_value = float(eer_value)
                print(f"{map_model_names(model)}, {eer_value:.4f}")
            print()
    else:
        print("Invalid JSON data")

# Load JSON data from file
file_path = "results.json"  # Change this to the actual path of your JSON file

with open(file_path, "r") as file:
    json_data = json.load(file)
    print_eer_values(json_data)
# except FileNotFoundError:
#     print(f"File not found: {file_path}")
# except json.JSONDecodeError:
#     print(f"Invalid JSON format in file: {file_path}")
# except Exception as e:
#     print(f"An error occurred: {e}")
