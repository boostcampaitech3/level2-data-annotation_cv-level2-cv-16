for i in ch8_training_*
do
  unzip $i -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_training_images
done
unzip ch8_training_localization_transcription_gt_v2.zip -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_training_gt