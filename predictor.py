from data_prep import RecommenderDataPrep

def main():
    print("Loading and preparing data...")
    data_prep = RecommenderDataPrep()
    data_prep.load_and_prepare()

if __name__ == "__main__":
    main()