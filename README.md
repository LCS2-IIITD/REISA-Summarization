# Multi-Document Summarization using Selective Attention Span and Reinforcement Learning

## Requirements

### Installation
Create a virtual environment and install dependencies:
```bash
git clone https://github.com/LCS2-IIITD/REISA-Summarization.git
cd REISA-Summarization/src

virtualenv -p python3.6 venv
source venv/bin/activate

# Install the according versions of torch and torchvision
pip install -r requirements.txt

```

### Dataset
Download the datasets -- [CQASumm](https://bitbucket.org/tanya14109/cqasumm/) or [Multinews](https://github.com/Alex-Fabbri/Multi-News).
Copy the dataset under the ```Data/``` directory and process the ```.txt``` files to ```.bin``` files and run ```python preprocess_data.py``` (Refer https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py).  

## Training and Evaluation

### Running code
Configure datasets folder as per training requirements and run
```bash
python train.py
```

### Evaluation
The checkpoints will be saved in checkpoints folder. For evaluation, run
```bash
python eval.py
```

```src``` contains the source code. ```semantic.csv``` contains the loss values.