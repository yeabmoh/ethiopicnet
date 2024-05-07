# Amharic Character Recognition Project

## Overview
This project is focused on evaluating the effectiveness of various Convolutional Neural Network (CNN) architectures in recognizing handwritten Amharic characters. The models tested include AlexNet, VGGNet, ResNet, and DenseNet, with an emphasis on their ability to adapt to the complexities of the Amharic script, which comprises 33 base characters each modified into 231 principal characters due to various vowel forms. The project utilizes the [Handwritten Amharic Character Dataset](https://github.com/Fetulhak/Handwritten-Amharic-character-Dataset), which provides a substantial foundation for training and testing the models.

The primary goal is to assess each model's performance across different metrics, particularly focusing on how well they generalize to new, unseen data — a critical factor for practical applications. Additionally, the project explores domain generalization techniques to enhance model robustness against domain shifts commonly encountered in real-world settings.

This project was written as part of a semester long class neuro140: Artificial and Biological intelligence offered by Gabriel Krieman.
## Repository Contents

- **am_dataset/** - Contains the dataset used for training and testing the models, sourced from the Handwritten Amharic Character Dataset.
- **mistakes/** - Stores data of misclassifications made by each model during testing.
- **newDomain_results/** - Includes results and performance metrics when the models are tested on a newly created domain to assess generalizability.
- **training_results/** - Contains the results from the initial training phase of the models on the standard dataset.
- **NotoSerif.ttf** - The TrueType font file used for any text rendering needs in plots or graphics.
- **ethiopic.ipynb** - A Jupyter notebook containing the main script for training and evaluating the CNN models.
- **ethiopic.py** - A Python script version of the Jupyter notebook for running in environments without Jupyter support.
- **ethiopic_newDS.ipynb** - Notebook used for testing the models on a new, unseen dataset.
- **mistakes_generator.ipynb** - Generates a detailed report on the mistakes made by each model.
- **mistakes_list.csv** - A CSV file listing all the mistakes identified by each model.
- **supported_chars.csv** - A CSV file listing all Amharic characters supported by the models.
- **visualize_results.ipynb** - Notebook for visualizing training and testing results through various plots and graphs.

## Usage
To use this repository, clone it locally and navigate to the desired notebook or script. Ensure you have the necessary Python libraries installed. Each script and notebook is self-contained and provides specific functionality as outlined above.

## Contributing
Contributions to this project are welcome as it is an ongoing academic endeavor!


## Just for Fun: A Little Poem by yours truly (chatGPT)

In the matrix of dots and lines,
Where characters hide in digital fines,
A quest unfolds with neural might,
To teach machines to read them right.

Amharic glyphs of ancient lore,
Now meet algorithms that adore,
Each curve and stroke they dare to see,
Transcribed by code, so deftly free.

Four models dance, a byte-filled ball,
DenseNet and ResNet, giants tall,
Alex and VGG join the fray,
In pixel fields, they make their play.

Mistakes are made, lessons learned,
Through epochs all, as tensors turned.
A dance of data, joy and strife,
This learning slice, a coder’s life.

So here we end, our journey's stop,
Feel free to tweak, to swap or chop.
In open source, we trust and share,
Your contributions show you care.

