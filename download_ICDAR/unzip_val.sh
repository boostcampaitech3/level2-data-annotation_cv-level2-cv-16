for i in ch8_validation_*
do
  unzip $i -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_validation_images
done
unzip ch8_validation_localization_transcription_gt_v2.zip -d /opt/ml/input/data/ICDAR17_MLT/raw/ch8_validation_gt