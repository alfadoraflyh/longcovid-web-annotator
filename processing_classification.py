import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder

# Set device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer (using bert-base-cased as in your code)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

le = LabelEncoder()
le.fit([-1, 0, 1])

# Define the model architecture (BERTFineTuned)
class BERTFineTuned(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(BERTFineTuned, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)  # Adjust output dimension if needed

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

# Initialize the model with your desired dropout rate
model = BERTFineTuned(dropout_rate=0.4)
# Load the saved model weights. Ensure 'BERT_best_model.pth' is available in the same directory.
model.load_state_dict(torch.load('model/BERT_best_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()  # Set model to evaluation mode

def encode_custom_texts(custom_texts, tokenizer, max_length):
    encoded_inputs = tokenizer(
        custom_texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True
    )
    return encoded_inputs

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove newlines and extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # apply function hashtag normalizer
    # text = re.sub(r'#\w+', hashtag_normalizer, text()
    # lowercase text
    text = text.lower()
    return text

def predict_text(custom_texts, max_length=182):
    """
    Tokenize the input texts, run the model and return the predicted classes.
    """
    custom_text_clean = clean_text(custom_texts)
    encoded = encode_custom_texts([custom_text_clean], tokenizer, max_length)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
    # Since there is only one text, extract the first prediction
    predicted_class = le.inverse_transform(predicted_labels)[0]
    return predicted_class