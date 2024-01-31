# Detect-AI-Generated-Essays

![AI vs Human](./images/human_vs_ai_essay.png)

## Description
Detect-AI-Generated-Essays is a machine learning tool aimed at helping instructors determine the authenticity of student-submitted essays. In an educational landscape where AI-generated content is becoming increasingly indistinguishable from human writing, this tool serves as a crucial asset for maintaining academic integrity.

## Motivation
The motivation for this project stems from the need to preserve trust and authenticity in academic work. As AI writing tools become more accessible, there is a growing concern that such technology could be misused. This project is driven by the belief that educators deserve robust tools to verify the originality of their students' work.

## Features
- **Advanced Detection**: Utilizes the 'deberta-v3-base' transformer model architecture for high accuracy in distinguishing between human and AI-generated text.
- **Comprehensive Pretraining**: The model is pretrained on a dataset containing 500k pairs of general texts, ensuring it understands a wide range of writing styles.
- **Interactive Web App**: Experience the AI essay detector firsthand through this [interactive Web App](https://detect-ai-generated-essays.streamlit.app/). This user-friendly interface allows for easy interaction with the tool.

## Technologies Used
This project was developed using a high-end GPU (A6000 with 48 GB of RAM) to ensure efficient processing of large datasets and rapid execution of the model.

## Installation and Usage

To reproduce the results of the Detect-AI-Generated-Essays tool, follow these steps carefully:

1. Clone the GitHub repository:
    ```
    git clone https://github.com/Anas-Badaoui/Detect-AI-Generated-Essays.git
    ```
2. Navigate to the project directory:
    ```
    cd Detect-AI-Generated-Essays
    ```
3. Create and activate a new virtual environment:
    ```
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On Unix or MacOS
    source venv/bin/activate
    ```
4. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
5. Download the large dataset used for pretraining from the Kaggle dataset:
   - Access the dataset from this link [here](https://www.kaggle.com/datasets/anas97/detect-ai-generated-essays-input-data)
   - Place the downloaded files into the `data` folder within the project directory.

6. Run the pretraining script:
    ```
    python pretrain_model.py
    ```

7. Open and run the `train.ipynb` Jupyter notebook to train the model.

### Disclaimers and Recommendations
- The input data is approximately 6GB in size, which may take some time to download and load locally.
- The pretraining and training were conducted on a high-end GPU (A6000 with 48 GB of RAM). If you encounter an out-of-memory error, it is likely due to using a GPU with less memory. In that case, consider reducing the batch size in the training script to accommodate the capabilities of your hardware.

## Credits and Acknowledgments
Credits to Guanshuo XU for creating the large dataset that was instrumental in the pretraining phase of this project. Their contribution is greatly appreciated and recognized. [Guanshuo XU](https://www.kaggle.com/wowfattie)

## License
This project is made available under the APACHE 2.0 License. See the [LICENSE](LICENSE) file for more information.
