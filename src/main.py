from src.train_models import train_all_models

if __name__ == "__main__":
    print("Starting model training...")
    train_all_models()
    print("Training completed. Models saved in /models.")
