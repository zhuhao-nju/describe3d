# High-fidelity 3D Face Generation from Natural Language Descriptions

**[Updates]**

- 2023.05.04: Release the inference code and pre-trained model.

**[To do list]**

- Release Describe3D Dataset

## Getting started

#### Requirements

- Python = 3.8
- pytorch = 1.7.1
- cudatoolkit = 11.0

#### Configure the environment

1. First, you need to build the virtual environment.

   ```
   conda create -n describe3d python=3.8
   ```

2. Then, you need to install CLIP. Please refer to https://github.com/openai/CLIP

3. Install other dependencies.

   ```
   pip install -r requirements.txt
   ```

#### Usage

1. Download the pre-trained texture generation model and put it into the checkpoints/texture_synthesis/

   https://drive.google.com/drive/folders/1zqCLaF-KzhWy_YSMqKf15aEKiv19lXz5?usp=sharing

2. Then you can run the main.py to generate 3D Faces.

```
python main.py --name="your model name" --descriptions="the description of the face you want to generate." --prompt="the abstract descriptions"

## Here are some examples we showed in the result folder.

python main.py --name="Stark" --descriptions="This middle-aged man is a westerner. He has big and black eyes with the double eyelid. He has a medium-sized nose with a high nose bridge. His face is square and medium. He has a dense and black beard." --prompt="Tony Stark."

python main.py --name="food" --descriptions="This young man is a westerner. His face is long and thin. He has big and round eyes. His nose is big with a high nose bridge. He has medium width mouth. He has no beard." --prompt="a man ate too much unhealthy food"

python main.py --name="old" --descriptions="This young man is a westerner. His face is oval and medium. He has big and round eyes. His nose is medium-sized with a high nose bridge. He has no beard." --prompt="He is a grandfather." 

python main.py --name="makeup" --descriptions="This girl is Asian. She has small and slender eyes with single eyelid. Her nose is small and wide. Her face is heart-shaped and medium." --prompt="a girl with makeup" 

```

