## Data

This folder contains only **part** of the data needed to run these scripts. The majority of the VGG embeddings need to be downloaded separately as we cannot host them in Github due to size.

Download the embeddings from FigShare: link. Please unzip the folders in this directory (`data`). There should be a total of three folders: `sketchpad_basic_fixedpose96_high`, `sketchpad_basic_fixedpose96_mid`, and `sketchpad_basic_fixedpose96_early`.

We provide a brief description of the files in this folder. As a user, you do not need to touch any of these.

    - `sketchpad_basic_fixedpose96_high`: embeddings from VGG FC6 layer for all sketches and photos.
    - `sketchpad_basic_fixedpose96_mid`: embeddings from VGG CONV-4-2 layer for all sketches and photos.
    - `sketchpad_basic_fixedpose96_early`: embeddings from VGG POOL-1 layer for all sketches and photos.
    - `human_confusion.npy`: a confusion matrix of human annotations matching sketches and photos. This numpy array contains aggregate statistics.
    - `human_confusion_object_order.csv`: labels for column order in human_confusion.npy.
    - `incorrect_trial_path_pilot2.txt`: certain game ids to ignore.
    - `invalid_trial_paths_pilot2.txt`: more game ids to ignore.
    - `sketchpad_label_dict.pickle`: map from game id to object class.
    - `sketchpad_context_dict.pickle`: map from game id to context.
