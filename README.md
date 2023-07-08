# Skincare API

The Skincare API is a Python-based web application developed using Django, PostgreSQL, and Docker. It provides functionality to analyze skin subjects by processing facial features extracted from user-provided photos. The API offers several key features, including acne detection, skin age estimation, beauty number calculation, and skin ethnicity detection. Each of these detections is performed using different intelligent models trained on relevant datasets, and the exported models are utilized for performing the tasks.

Please note that the trained models are excluded from this project in accordance with the policies of Raman Pardaz Gharb company.

## Key Features

The Skincare API offers the following key features:

- **Acne Detection**: Utilizes the Fast R-CNN ResNet-50 model based on PyTorch for accurate acne detection on facial images.
- **Skin Age Estimation**: Employs the DeepFace model to estimate the age of the individual based on their facial features.
- **Beauty Number Calculation**: Utilizes the Combonet network to calculate a beauty number ranging from 0 to 5, indicating the degree of beauty.
- **Skin Ethnicity Detection**: Detects the ethnicity of the individual by analyzing their facial features.

## Contributing

Contributions to the Skincare API project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m "Add new feature"`
4. Push your changes to the branch: `git push origin feature-branch`
5. Submit a pull request detailing your changes.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

