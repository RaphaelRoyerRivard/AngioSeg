# find data/Angio_Ped/img -name "*.jpg" | awk '{split($1,tab, "/"); print "python inference_image.py -i "$1" -d G:/RECHERCHE/Work_CORSTEM/data/Angio_Ped"}' 
dirout=G:/RECHERCHE/Work_CORSTEM/testoutputbatch


name=output
inference=inference_image_fundus.py
model=FantinNet_fundus.py
checkpoint=G:/RECHERCHE/Work_DL/DATA_HDD/alldata/checkpoint256_new_3labels_deconv_20170725_3d_enh/model_112000_91.8762.ckpt-112000
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/inferAngioV00.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v00/test/img
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v00/train/img
mkdir $dirout/AngioNet_v00/test/$name
mkdir $dirout/AngioNet_v00/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); print "python inference_image.py -i "pin[1]" -c '$checkpoint' -d '$dirout'/AngioNet_v00/test"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, ".");  print "python inference_image.py -i "pin[1]" -c '$checkpoint' -d '$dirout'/AngioNet_v00/train"}' >> $scriptout

name=output_enh
inference=inference_image_enh.py
model=FantinNetAngio_deconv.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/checkpoint/model_7600_96.3792.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/inferAngioV1_enh.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/test/temp/seg/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/train/temp/seg/
mkdir $dirout/AngioNet_v1/test/$name
mkdir $dirout/AngioNet_v1/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/test/'$name'/"pin1[1]".png"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/train/'$name'/"pin1[1]".png"}' >> $scriptout

name=output_enh_100e
inference=inference_image_enh.py
model=FantinNetAngio_deconv.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/checkpoint/model_76800_95.8889.ckpt-76800
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/inferAngioV1_enh_100e.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/test/temp/seg/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/train/temp/seg/
mkdir $dirout/AngioNet_v1/test/$name
mkdir $dirout/AngioNet_v1/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/test/'$name'/"pin1[1]".png"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/train/'$name'/"pin1[1]".png"}' >> $scriptout

name=output_2d
inference=inference_image_2d.py
model=FantinNetAngio_deconv.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/checkpoint_2d/model_7600_96.2987.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/inferAngioV1_2d.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/test/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/train/
mkdir $dirout/AngioNet_v1/test/$name
mkdir $dirout/AngioNet_v1/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.tif" ! -wholename "*seg*"| awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/test/'$name'/"pin[1]".png -e '$imgintest'/temp/seg/"pin[1]"_enh.jpg"}' >> $scriptout
find $imgintrain -name "*.tif" ! -wholename "*seg*" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, ".");  print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/train/'$name'/"pin[1]".png -e '$imgintrain'/temp/seg/"pin[1]"_enh.jpg"}' >> $scriptout

name=output
inference=inference_image_ori.py
model=FantinNetAngio_deconv.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v0/checkpoint/model_7600_96.2717.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/inferAngioV0.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v0/test/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v0/train/
mkdir $dirout/AngioNet_v0/test/$name
mkdir $dirout/AngioNet_v0/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.tif" ! -wholename "*seg*" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v0/test/'$name'/"pin[1]".png"}' >> $scriptout
find $imgintrain -name "*.tif" ! -wholename "*seg*" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, ".");  print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v0/train/'$name'/"pin[1]".png"}' >> $scriptout

name=output_enh_up
inference=inference_image_enh.py
model=FantinNetAngio_upscale.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/checkpoint_upscale/model_7600_96.6661.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/inferAngioV1_enh_up.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/test/temp/seg/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/AngioNet_v1/train/temp/seg/
mkdir $dirout/AngioNet_v1/test/$name
mkdir $dirout/AngioNet_v1/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/test/'$name'/"pin1[1]".png"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/AngioNet_v1/train/'$name'/"pin1[1]".png"}' >> $scriptout

name=output_clahe
inference=inference_image_enh.py
folder=AngioNet_v2
model=FantinNetAngio_upscale.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/$folder/checkpoint/model_7600_96.5708.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/infer${folder}_$name.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/$folder/test/temp/seg/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/$folder/train/temp/seg/
mkdir $dirout/$folder/test/$name
mkdir $dirout/$folder/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/'$folder'/test/'$name'/"pin1[1]".png"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/'$folder'/train/'$name'/"pin1[1]".png"}' >> $scriptout

name=output_tophat
inference=inference_image_enh.py
folder=AngioNet_v3
model=FantinNetAngio_upscale.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/$folder/checkpoint/model_7600_93.4829.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/infer${folder}_$name.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/$folder/test/temp/seg/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/$folder/train/temp/seg/
mkdir $dirout/$folder/test/$name
mkdir $dirout/$folder/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/'$folder'/test/'$name'/"pin1[1]".png"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/'$folder'/train/'$name'/"pin1[1]".png"}' >> $scriptout

name=output_tophat_ms
inference=inference_image_enh_3c.py
folder=AngioNet_v4
model=FantinNetAngio_upscale.py
checkpoint=G:/RECHERCHE/Work_CORSTEM/data/$folder/checkpoint/model_7600_96.4469.ckpt-7600
scriptout=G:/RECHERCHE/Work_CORSTEM/trunk/script/infer${folder}_$name.bat
imgintest=G:/RECHERCHE/Work_CORSTEM/data/$folder/test/temp/seg/
imgintrain=G:/RECHERCHE/Work_CORSTEM/data/$folder/train/temp/seg/
mkdir $dirout/$folder/test/$name
mkdir $dirout/$folder/train/$name
echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model" FantinNetAngio.py" >> $scriptout
find $imgintest -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/'$folder'/test/'$name'/"pin1[1]".png"}' >> $scriptout
find $imgintrain -name "*.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$dirout'/'$folder'/train/'$name'/"pin1[1]".png"}' >> $scriptout

