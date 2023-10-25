# Emotion Detection Model

This repository contains the code and model for our Emotion Detection service, which predicts user emotions from audio input.

## Table of Contents
- [Description](#description)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Description

Our Emotion Detection model uses machine learning to predict emotions from audio input. It's designed to help users analyze their emotional state and is integrated with our web application.

## Deployment

We recommend deploying the model on Hugging Face's Model Hub. Here are the steps:

1. **Upload Your Model:**
   - Fork this repository and make your changes if necessary.
   - Commit your changes and push to your repository.
   - Go to the Hugging Face Model Hub (https://huggingface.co/models) and click "New Model."
   - Enter your repository's URL and details to create the model.

2. **Create an Inference API:**
   - Go to the "Inference API" tab in your Hugging Face account.
   - Configure the API with the relevant details (e.g., environment, GPU, CPU).
   - By now You would have ML endpoint that can be used in your front end.

3. **Usage from the Front End:**
   - To use the API endpoint in your web application, refer to the front-end documentation:
     [front-end-emotion-detection](https://github.com/Psych-2-Go-Ai/front-end-emotion-detection)

## Contributing

We welcome contributions and feedback from the community. If you find any issues or have suggestions, please open an issue or a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
