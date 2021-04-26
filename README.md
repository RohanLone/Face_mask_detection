# Face_mask_detection
Using Tranfer learning, trained this model to detect the human faces with masks and without masks.

In this project, I used Google Colab (for model training), Google Drive (for storage) and Local machine (for model testing).
Converted XML to CSV using [xml_to_csv.py](https://github.com/RohanLone/Tensorflow_Object_Detection_with_Tensorflow_2.0/blob/main/xml_to_csv.py) script. 
This script [generate_tfrecords.py](https://github.com/RohanLone/Tensorflow_Object_Detection_with_Tensorflow_2.0/blob/main/generate_tfrecord.py) will be used to covert the csv into the TFRecord format. 

#### Result
![alt-text](https://github.com/RohanLone/Face_mask_detection/blob/main/doc/result.png)


1. 50 Images are used for model building. Used LabelImg for labelling the images (Output=XMLFile)
2. Split 90% and 10% data for Train and Test respectively.
3. Generated TF Records from these splits
4. configured a .config file for the model (ssd_mobilenet_v2_320x320_coco17_tpu-8) as follows:
  * num_classes: 2 (1. Mask , 2. Without_mask)
  * fine_tune_checkpoint: "ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
  * fine_tune_checkpoint_type: "detection"
  * batch_size: 4
  * learning_rate: .0001
 
 ```
   - train_input_reader: {
     label_map_path: "images/labelmap.pbtxt"
     tf_record_input_reader {
     input_path: "train.record" }}
 
   - eval_input_reader: {
     label_map_path: "images/labelmap.pbtxt"
     shuffle: false
     num_epochs: 1
     tf_record_input_reader {
     input_path: "test.record"}}
  ```
5. Trained on Google Colab
6. Exported graph from new trained model
7. Detect on new data!

#### Project structure

![alt-text](https://github.com/RohanLone/Face_mask_detection/blob/main/doc/Project%20Structure.png)



#### Performance of model

![alt-text](https://github.com/RohanLone/Face_mask_detection/blob/main/doc/Performance%20of%20model.png)
* Average Precision: 0.878
* Average Recall: 0.883
* Mean Average Precision: 0.878218
* Mean Average Recall: 0.883333
* Total Loss: 0.180401


#### Commands to run
Training:
```
From within object_detection:
!python model_main_tf2.py --pipeline_config_path=training/ssd_mobilenet_v2_320x320_coco17_tpu8.config --model_dir=training --alsologtostderr
```
To download Inference Graph:
```
!python exporter_main_v2.py --trained_checkpoint_dir=training --pipeline_config_path=training/ssd_mobilenet_v2_320x320_coco17_tpu-8.config --output_directory inference_graph
```


