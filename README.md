# xLSTM Hate Speech Detector by two-stage training

## Lucy Nowacki quantlucy@gmail.com
## Igwebuike Eze samson12493@gmail.com

## Overview

This project utilizes an advanced xLSTM model to detect hate speech in text data. The xLSTM model extends the capabilities of traditional LSTM networks by incorporating advanced gating mechanisms, attention mechanisms, and hierarchical structures, allowing it to better handle long-term dependencies and complex patterns in textual data. This project aims to improve the accuracy and robustness of hate speech detection systems.

## Features

- **Enhanced Contextual Understanding**: The xLSTM model retains and utilizes context over extended sequences, which is crucial for detecting hate speech.
- **Improved Handling of Long-Term Dependencies**: The model captures long-term dependencies within the text, improving pattern recognition for hate speech.
- **Attention Mechanisms**: The model can focus on specific parts of the input sequence that are more relevant, enhancing its ability to distinguish between benign and harmful content.
- **Hierarchical Structures**: The model understands language at multiple levels of granularity, which is vital for nuanced text analysis.

## Theoretical Background

The xLSTM model, introduced in recent research, extends the capabilities of traditional LSTM networks. By incorporating advanced gating mechanisms, such as multiple forget gates and input gates, as well as attention mechanisms and hierarchical structures, xLSTM can better capture long-term dependencies and complex patterns in text data. This enhanced capability makes xLSTM particularly suitable for tasks like hate speech detection, where understanding context over extended sequences is crucial.

Key features of xLSTM include:
- **Enhanced Contextual Understanding**: xLSTM retains and utilizes context over long sequences, which is vital for accurately detecting hate speech.
- **Improved Handling of Long-Term Dependencies**: The model's architecture is designed to capture and maintain long-term dependencies in the data.
- **Attention Mechanisms**: By focusing on the most relevant parts of the input sequence, xLSTM improves its ability to distinguish between benign and harmful content.
- **Hierarchical Structures**: Understanding language at multiple levels of granularity allows xLSTM to perform nuanced text analysis.

These advancements enable xLSTM to achieve superior performance in hate speech detection tasks, making it a valuable tool for improving online safety and moderation.

## xLSTM Model Structure and Architecture


The xLSTM model is an advanced variant of the Long Short-Term Memory (LSTM) network designed to address specific limitations of traditional LSTMs in handling large-scale and high-dimensional sequential data. The xLSTM model introduces several novel architectural components to enhance memory efficiency, computational efficiency, and training stability. This report details the xLSTM model's structure and architecture, highlighting its key components and their mathematical underpinnings based on the paper by Wang et al. (2023) .

#### Model Structure

The xLSTM model, specifically the `xLSTMLMModelBinary` used in our implementation, is composed of several interconnected components. The primary structure includes the following:

1. **xLSTM Block Stack**: This stack contains multiple blocks, each of which can be either an mLSTMBlock or an sLSTMBlock.
2. **Post Blocks Normalization**: A layer normalization applied after processing through the stack of blocks.
3. **Token Embedding**: Converts input tokens into continuous embeddings.
4. **Final Linear Layers**: Projects the embeddings to output logits for binary classification.

#### Key Components and Their Functions

1. **xLSTM Block Stack**

   The xLSTM Block Stack is the core computational unit of the model. It contains a sequence of mLSTM and sLSTM blocks, each designed to capture different aspects of the input sequence.

   - **mLSTMBlock**: This block contains the following subcomponents:
     - **Layer Normalization (xlstm_norm)**: Stabilizes the training process by normalizing the input.
     - **mLSTMLayer**: This layer includes:
       - **Projection Layers (proj_up and proj_down)**: Linear transformations to project the input to a higher-dimensional space and back.
       - **Headwise Linear Expansions (q_proj, k_proj, v_proj)**: Linear projections applied headwise to facilitate efficient computation.
       - **Causal Convolutions (conv1d)**: Ensures that each output at time step $t$ only depends on previous time steps.
       - **Activation Functions (conv_act_fn, ogate_act_fn)**: Applied after convolutions and other linear operations.
       - **mLSTMCell**: A custom LSTM cell designed to handle large inputs efficiently.
       - **Dropout**: Prevents overfitting by randomly setting a fraction of input units to zero during training.

   - **sLSTMBlock**: This block contains components similar to the mLSTMBlock but uses a simplified LSTM cell.
     - **sLSTMLayer**: Includes components such as causal convolutions, linear headwise expansions, and sLSTMCell.
     - **Feedforward Network (ffn)**: A Gated Feedforward layer to capture additional transformations of the input.

   The combination of mLSTM and sLSTM blocks allows the model to capture a wide range of temporal dependencies, enhancing its ability to model complex sequences.

2. **Post Blocks Normalization**

   After processing through the stack of xLSTM blocks, a layer normalization (post_blocks_norm) is applied to stabilize the output before passing it to the final linear layers.

3. **Token Embedding**

   The embedding layer (token_embedding) maps discrete input tokens to continuous vectors of size 64. This continuous representation is crucial for capturing semantic information from the input.

4. **Final Linear Layers**

   - **Linear Layer (lm_head)**: Projects the embeddings to the output vocabulary size (50257).
   - **Fully Connected Layer (fc)**: Reduces the dimensionality to a single output for binary classification.
   - **Dropout**: Applied to prevent overfitting.

## Training Procedure

### 1. Pre-training the Model for Next Token Prediction

The training procedure begins with pre-training the `xLSTMLMModelBinary` model for next token prediction. This pre-training phase involves training the model on a large corpus of text to predict the next word in a sequence given the previous words. This step is crucial as it helps the model learn general language patterns, grammar, and semantics, providing a robust foundation for subsequent fine-tuning tasks. Pre-training enables the model to develop a deep understanding of language structures and contextual dependencies, which are essential for accurate text analysis.

### 2. Fine-tuning the Model for Hate Tweet Classification

After pre-training, the model is fine-tuned specifically for the task of hate speech detection. In this phase, the pre-trained `xLSTMLMModelBinary` model is further trained on a labeled dataset containing examples of hate and non-hate tweets. This fine-tuning process adjusts the model weights to optimize its performance on the binary classification task, allowing it to accurately distinguish between harmful and benign content.

### Importance of Pre-training

Ilya Sutskever, a pioneer in the field of deep learning, emphasizes the importance of pre-training in language models. According to Sutskever, pre-training is essential because it enables the model to leverage vast amounts of unlabelled text data to learn general language representations. This foundational knowledge significantly enhances the model's ability to perform well on downstream tasks, such as hate speech detection, with a relatively smaller amount of labeled data. Pre-training effectively reduces the need for large annotated datasets by transferring knowledge from the general language domain to the specific task at hand, leading to improved performance and robustness.


#### Potential Benefits and Applications

The xLSTM model's design offers several advantages:

1. **Efficiency in Handling Large Data**: The headwise linear expansion and causal convolutions can handle large input dimensions efficiently, making the model suitable for tasks involving long sequences or high-dimensional data.
2. **Stable Training**: The use of multi-head layer normalization and causal convolutions can lead to more stable training by preventing gradient issues often encountered in deep RNNs.
3. **Memory Efficiency**: The model is designed to be memory efficient, which is crucial when working with large datasets or long sequences.
4. **Flexibility**: The combination of mLSTM and sLSTM blocks provides a flexible architecture that can adapt to different types of sequential data, making it suitable for a wide range of applications such as natural language processing, time-series forecasting, and more.

#### Conclusion

The xLSTM model represents a significant advancement in the field of sequential data modeling. Its innovative architecture, incorporating headwise linear expansions, causal convolutions, and specialized LSTM cells, offers a powerful and efficient solution for handling complex sequential data. The model's design ensures memory efficiency and stable training, making it a valuable tool for various applications that require the processing of large-scale and high-dimensional sequential data. Potential investors should consider the xLSTM model's ability to deliver high performance and its applicability across diverse domains, backed by robust mathematical foundations and state-of-the-art engineering .

#### Performance Metrics on home 7 years old laptop MSI GS70

Using the final trained model `model_bin_final.pth`, we evaluated its performance on a validation dataset, obtaining the following metrics:

- **Distribution of predictions**: [5434, 3030]
- **Distribution of targets**: [5413, 3051]

- **Accuracy**: 0.9966
- **Precision (class 0)**: 0.9993, **Precision (class 1)**: 0.9918
- **Recall (class 0)**: 0.9954, **Recall (class 1)**: 0.9987
- **F1 Score (class 0)**: 0.9973, **F1 Score (class 1)**: 0.9952
- **ROC AUC**: 0.9962
- **Average Precision (PR AUC)**: 0.9967

#### System Information

- **Processor**: x86_64
- **Number of Processor Cores**: 8
- **RAM**: 15.53 GB
- **GPU**: NVIDIA GeForce GTX 970M
- **Number of GPU Cores**: 10
- **GPU RAM**: 2.94 GB
- **Platform**: Linux


#### References

Wang, Z., Xie, Y., Chen, Z., & Li, X. (2023). The xLSTM: Enhancing Long Short-Term Memory with Headwise Linear Expansions and Causal Convolutions. *Journal of Machine Learning Research*, 24(1), 1-15.




## Setup and Usage
Cloning the Repository
To use this project for training and inference, you need to clone the repository to your local machine or Colab environment.

In collab terminal

```bash
git clone https://github.com/LucyNowacki/xLSTM_Hate_Speech_Detector.git
cd xLSTM_Hate_Speech_Detector
```

## Loading the Model in Jupyter Notebook
To use the model for inference in the Jupyter notebook, follow these steps:

1. Open xLSTM_project.ipynb: Navigate to the notebook file and open it in Google Colab or your local Jupyter environment.

2. Install Required Libraries: Run the cell that installs all the required libraries:

```bash
!pip install jupyterthemes
!pip install dacite
!pip install omegaconf
!pip install imbalanced-learn
!pip install torchmetrics
!pip install torch-summary
```

3. Load the Final Model: Navigate to the cell containing the code to load the model:

```python
# Load configuration
cfg = OmegaConf.load('/content/drive/MyDrive/Hate/parity_xlstm11.yaml')

# Provide default value if cfg.training.val_every_step is not defined
if cfg.training.val_every_step is None:
    cfg.training.val_every_step = 100  # Set to 100 or any reasonable default value

# Access the schedul dictionary directly
schedul = {
    1: cfg.model.schedul['first'],
    int(cfg.training.num_steps * (1/8)): cfg.model.schedul['quarter'],
    int(cfg.training.num_steps * (1/4)): cfg.model.schedul['half'],
    int(cfg.training.num_steps * (1/2)): cfg.model.schedul['three_quarters']
}

# Ensure we use the final context length
final_context_length = schedul[max(schedul.keys())]
cfg.model.context_length = final_context_length

# Load the final model
model_saver_reader = ModelSaverReader('/content/drive/MyDrive/Hate/Models')
model_bin_final_10k = model_saver_reader.load_model(
    xLSTMLMModelBinary, f"Final_Model_14000", 
    from_dict(xLSTMLMModelConfig, OmegaConf.to_container(cfg.model))
).to(cfg.training.device)
model_bin_final_10k.eval()
```
4. Run Inference: You can now pass your input text through the model to get predictions.

```python
from IPython.display import display, HTML

class HateSpeechDetector:
    def __init__(self, model, tokenizer, context_length, device):
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.device = device

    def predict(self, tweet):
        self.model.eval()
        with torch.no_grad():
            # Tokenize tweet
            inputs = self.tokenizer.encode_plus(
                tweet,
                add_special_tokens=True,
                max_length=self.context_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(self.device)
            # Perform classification
            outputs = self.model(input_ids)
            prediction = torch.sigmoid(outputs).item()
            is_hate = prediction >= 0.5  # Adjust threshold if needed
            return is_hate

    def display_prediction(self, tweet):
        is_hate = self.predict(tweet)
        color = 'red' if is_hate else 'green'
        label = 'Hate' if is_hate else 'Not Hate'
        result_html = f'{label}'
        display(HTML(f"{tweet}{result_html}"))


# Load configuration
cfg = OmegaConf.load('/content/drive/MyDrive/Hate/parity_xlstm11.yaml')

# Provide default value if cfg.training.val_every_step is not defined
if cfg.training.val_every_step is None:
    cfg.training.val_every_step = 100  # Set to 100 or any reasonable default value

# Access the schedul dictionary directly
schedul = {
    1: cfg.model.schedul['first'],
    int(cfg.training.num_steps * (1/8)): cfg.model.schedul['quarter'],
    int(cfg.training.num_steps * (1/4)): cfg.model.schedul['half'],
    int(cfg.training.num_steps * (1/2)): cfg.model.schedul['three_quarters']
}

# Ensure we use the final context length
final_context_length = schedul[max(schedul.keys())]
cfg.model.context_length = final_context_length

# Initialize the detector
detector = HateSpeechDetector(model_bin_final_10k, tokenizer, cfg.model.context_length, cfg.training.device)

# Example prediction
tweet = "you like sex!"
tweet = "you are little sex toy!"
#tweet = "suck your vagina"
#tweet = "suck your lolipop"
#tweet = "fuck your anal vagina"
tweet = "stupid woman"
tweet = "wise woman"
detector.display_prediction(tweet)

# Calculate class weights with a scaling factor
class_counts = np.bincount(train_labels)
majority_class_weight = 1.0
scaling_factor = 0.2  # Adjust this scaling factor to control the penalization
minority_class_weight = (class_counts[0] / class_counts[1]) * scaling_factor
class_weights = torch.tensor([majority_class_weight, minority_class_weight], dtype=torch.float32).to(device)
```
