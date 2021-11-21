# Supermarket products

 
## Description   
This project shows how to fine-tune MaskRCNN. Add more images for it to generalize

## How to run   
First, install dependencies
```bash
# Make sure git lfs is installed and enabled
git lfs install

# clone project   
git clone https://github.com/fran6co/supermarket

# install project   
cd supermarket
conda env create -f environment.yml
conda activate supermarket

 ```   
 Next, you can train the network
 ```bash 
python project/instance_segmentation_detectron.py    
```
 Next, you can run the network on the validation images
 ```bash
python tests/test_detectron.py data/test.jpg --model_path=logs/version_0/model_final.pt
```

## Implementation details

### What are the challenges and your proposed solution?

The dataset is very small (2 images) and they are very similar to each other,
making validation a bit pointless given how similar the images are (hard to check for over-fitting and make sure the method is generalizing)

Development time is very reduced, as a side project it's hard to give it all the time and attention I would normally give to a problem.
Had to take decisions to reduce development time like using [Detectron2](https://github.com/facebookresearch/detectron2) and forgoing newer architectures like visual transformers that show [better results](https://github.com/microsoft/Swin-Transformer) and better deployability (detectron2 doesn't support exporting to ONNX, making it incompatible with [ONNXRuntime](https://onnxruntime.ai/))
You can find in `project/instance_segmentation_torchvision.py` a possible implementation using torchvision and [Pytorch Lightning](https://www.pytorchlightning.ai/) but it's not working properly.

A natural follow up to this project is to make it scale to more cases. Neural networks provide several advantages over other approaches as it can
- adapt to new data just by increasing the dataset and retraining
- augmentation can be added to improve the generalization of the net (shadows, lighting conditions, noisy camera, etc...)
- it can leverage pretrained neural networks by using transfer learning


### What’s the performance of your algorithm? How do you measure it?

The neural net can run at 2.6FPS for a 500x400 image on a Geforce 1650, i7-9750H. This was measured by just running the deployment version of it on a loop and measuring the mean of how long it took to run.

For the accuracy of the network I used tensorboard to check the accuracy for the reference image, given how similar it is to the test one it's a good tell on the performance:

![train_accuracy](assets/train_accuracy.png?raw=true "train_accuracy")

Then I did run the model on the test image and got the accuracy (how many mask bits are incorrect), 96% for the final model.

![output](assets/output.png?raw=true "output")

### Which annotation tool did you use? Explain why. 

[Label Studio](https://labelstud.io/) offers several advantages over other labeling tools:
- users don't need to download datasets in their computers to label
- all labeling is centralized in one place, easy to retrieve new labellings
- ability to distribute among multiple people label tasks
- can execute neural nets as part of the labeling, making it easier on labelers
- easy to setup

I used this to run it:
```bash
docker run -it -p 8080:8080 -v data:/label-studio/data heartexlabs/label-studio:latest
```

### If you choose to use open-source vision models, what are they? Explain why. 

I used MaskRCNN as it's a proven architecture for instance segmenation (and I have experience with it). [Detectron2](https://github.com/facebookresearch/detectron2) makes it really easy to fine-tune pretrained models and I knew it would converget to something useful given the small dataset.

### Can your solution generalize?  

Yes, it would require more data and possibly adding more augmentation to it. I went with this solution because it's meant to work on a bigger scope.

### Can your solution handle partial/full occlusion? How will you test it? 

It does handle partial occlusion, in the dataset there are several examples of it. MaskRCNN can handle it as long recognizable bits of the object are visible.

A proper test to measure how the solution handles occlusion would be block parts of objects and see when it stops seeing them. Another option is at labeling stage mark all the occluded products and then measure the accuracy on that subset.

Full occlusion would require making some assumptions about the shelf and distribution of the items. For example, normally items in shelfs are arranged in lines of the same type of product. So if we can only see one product and we can detect the position relative to the shelf. Then we can make guess the amount of occluded products by dividing the shelf depth by the product base dimension. Of course this is an over simplification, in reality products are arranged in less that ideal positions and different type of products can be mixed up. 

### Can your solution recognize the same set of products in a different store environment? What can be improved to increase your solution robustness?

It might, by transfer learning MaskRCNN it might have generalized a bit. To improve the robustness using only this data I would do:
 - Add more color augmentations to simulate different lighting conditions
 - Generate new backgrounds and create mashups with the segmented products (provided by the labeling)
 - Freezer layers in the neural network to reduce over-fitting
 - Add more data, maybe even downloading from the internet and labeling it
