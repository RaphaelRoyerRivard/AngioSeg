inference=$6 #
input=$1
model=$5

output=$7
preprocess=$2 # median or tophat or imageback
upscale=$3
trainv=$4
gt=$8
catin=$9 # catinvess or catinback
color=${10}
selection=${11}
grayscale=${12} #grayscale 1 or 0
scriptout=script.bat
postprocess=${13} #post value of elements to delete in pix
postprocesshyst=${14} #post value of elements to delete in pix
augm=${15}
thresholdsp=${16} # normally 100
resol=${17}

checkpoint=G:/RECHERCHE/Work_CORSTEM/data/checkpoints/${trainv}/${upscale}_${preprocess}_${color}_${selection}_${model}_${augm}/model_7600

mkdir $output

echo "python getenhanced.py "$input " " $output " " $preprocess " " $grayscale " " $resol > $scriptout
cmd /C "$scriptout"

echo "copy /Y inferences\\"$inference" inference_image.py" > $scriptout
echo "copy /Y models\\"$model".py FantinNetAngio.py" >> $scriptout
find $output -name "*.jpg" ! -name "*_back.jpg" | awk '{split($1,tab, "/"); split(tab[length(tab)],pin, "."); split(pin[1], pin1, "_enh"); print "python inference_image.py -i "$1" -c '$checkpoint' -o '$output'/"pin1[1]".png -r '$resol'"}' >> $scriptout
cmd /C "script.bat"

echo "python getROCcurve.py get " $input " " $output " " $gt " " $catin " " $postprocess " " $postprocesshyst " " $thresholdsp > $scriptout
cmd /C "script.bat"