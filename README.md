# Detect-AI-Generated-Essays

## Description
Detect-AI-Generated-Essays is a machine learning tool aimed at helping instructors determine the authenticity of student-submitted essays. In an educational landscape where AI-generated content is becoming increasingly indistinguishable from human writing, this tool serves as a crucial asset for maintaining academic integrity.

## Motivation
The motivation for this project stems from the need to preserve trust and authenticity in academic work. As AI writing tools become more accessible, there is a growing concern that such technology could be misused. This project is driven by the belief that educators deserve robust tools to verify the originality of their students' work.

## Features
- **Advanced Detection**: Utilizes the 'deberta-v3-base' transformer model architecture for high accuracy in distinguishing between human and AI-generated text.
- **Comprehensive Pretraining**: The model is pretrained on a dataset containing 500k pairs of general texts, ensuring it understands a wide range of writing styles.
- **Interactivity Coming Soon**: A web app will be deployed to allow users to interact with the AI essay detector in a user-friendly environment.

## Technologies Used
This project was developed using a high-end GPU (A6000 with 48 GB of RAM) to ensure efficient processing of large datasets and rapid execution of the model.

## Installation and Usage
To install and use Detect-AI-Generated-Essays, please follow these steps:

1. Clone the GitHub repository:
    ```
    git clone [Your GitHub Repo Link Here]
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

Further instructions on using the application will be provided once the web app deployment is complete.

## Credits and Acknowledgments
Heartfelt thanks to Guanshuo XU for creating the large dataset that was instrumental in the pretraining phase of this project. Their contribution is greatly appreciated and recognized. [Guanshuo XU](https://www.kaggle.com/wowfattie)

## License
This project is made available under the APACHE 2.0 License. See the [LICENSE](LICENSE) file for more information.
