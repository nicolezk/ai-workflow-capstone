from model import model_train, model_load

def main():
    
    model_train(test=False)

    model = model_load()
    
    print("model training complete.")


if __name__ == "__main__":

    main()
