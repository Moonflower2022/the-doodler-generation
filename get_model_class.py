import torch

if __name__ == '__main__':
    with open("models/latest_experiment.txt", "r") as file:
        model_path = file.read()

    info = torch.load(model_path)

    print(f"used model: {info["model_class_name"]}")