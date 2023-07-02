# Question Classifier Model using BERT+CNN

This repository contains the code for a question classifier model implemented using BERT (Bidirectional Encoder Representations from Transformers) and CNN (Convolutional Neural Network), using PyTorch.

The goal of the question classifier model is to classify input questions into different categories or classes. The model leverages the power of BERT, a state-of-the-art pre-trained language model, to capture the contextual information of the questions. The CNN component is added on top of BERT to extract local features and improve the classification performance.

## Requirements

To run the question classifier model, you need the following dependencies:

- Python 3.x
- PyTorch (>=1.6.0)
- Transformers (Hugging Face library, >=4.0.0)
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using pip:

```
pip install torch>=1.6.0 transformers>=4.0.0 numpy pandas scikit-learn
```

## Dataset

The model expects a labeled question dataset for training and evaluation. The dataset should be in CSV format with two columns: 'question' and 'label'. The 'question' column contains the input questions, and the 'label' column contains the corresponding class labels.

You need to place your dataset in the `data` directory. By default, the model assumes the dataset file name as 'questions.csv'. If your dataset has a different name or format, make sure to update the data loading code accordingly.

## Usage

To train and evaluate the question classifier model, follow these steps:

1. Prepare your labeled question dataset and place it in the `data` directory (as described in the Dataset section).

2. Run the `Notebook.ipynb` notebook:

   This script will perform the following steps:
   - Load and preprocess the dataset.
   - Split the dataset into training and validation sets.
   - Fine-tune the BERT model with the CNN layer on top using the training data.
   - Evaluate the model's performance on the validation set.
   - Save the trained model and evaluation results.

3. Once the training is complete, you can use the section provided in notebook classify new questions:
   
   Replace "Your question goes here" with the question you want to classify. The script will load the trained model and provide the predicted class label for the input question.

Feel free to modify the model architecture, hyperparameters, or training pipeline to suit your specific requirements.

## Acknowledgments

This question classifier model is based on the BERT and CNN architectures, inspired by various research papers and open-source implementations. The code for BERT integration is adapted from the Hugging Face Transformers library.

## License

This project is licensed under the MIT License. You can find the details in the `LICENSE` file.

If you use this code or find it helpful, I would appreciate a citation or acknowledgment.
