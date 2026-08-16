from datasets import load_dataset

DATASET_ID = "tarekmasryo/cancer-risk-factors-data"

def load_cancer_dataset():
    '''loading the cancer dataset from the Hugging Face Hub'''
    dataset = load_dataset(DATASET_ID)

    return dataset

if __name__=='__main__':
    dataset=load_cancer_dataset()

    print("\n Dataset ")
    print(dataset)

    print("\n Train Split ")
    print(dataset['train'])

    print('\n Dataset Features :')
    print(dataset['train'].features)

    print('\n Dataset columns: ')
    print(dataset['train'].column_names)

    print("\n Example Record: ")
    print(dataset['train'][2])