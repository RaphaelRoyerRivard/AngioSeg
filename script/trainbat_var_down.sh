enhanced=$1
selection=$2
model=$3
augm=$4
checkpointdir=E:/Fantin/AngioData/checkpoints/train70/upscale_${enhanced}_gray_${selection}_${model}_${augm}
numepoch=10
dirtrainin=E:/Fantin/AngioData/TRAIN_down
dirtestin=E:/Fantin/AngioData/TEST
patchdir=E:/Fantin/AngioData/TRAIN_down/patch/patch_train70_${enhanced}_${selection}
tfrecordtrain=E:/Fantin/AngioData/TRAIN_down/patch/patch_train70_${enhanced}_${selection}/tfrecordtrain.tfrecords
tfrecordtest=E:/Fantin/AngioData/TRAIN_down/patch/patch_train70_${enhanced}_${selection}/tfrecordtest.tfrecords
summarydir=E:/Fantin/AngioData/TRAIN_down/patch/summary/train70_${enhanced}_gray_${selection}_${model}_${augm}

echo "" | awk '{print "cp models/'$model'.py FantinNetAngio.py"}' |sh

mkdir $patchdir
mkdir $checkpointdir

#selection 1: rayonmin=10, rayonmax=50, space=10
patchsize=`echo " " | awk '{if ("'$selection'"=="selection4"){print 256} else if ("'$selection'"=="selection5"){print 64} else {print 128}}'`
rayonmin=`echo " " | awk '{if ("'$selection'"=="selection1"){print 10} else if ("'$selection'"=="selection2"){print 10} else if ("'$selection'"=="selection3"){print 10}else if ("'$selection'"=="selection4"){print 10}else if ("'$selection'"=="selection5"){print 5}}'`
rayonmax=`echo " " | awk '{if ("'$selection'"=="selection1"){print 50} else if ("'$selection'"=="selection2"){print 500} else if ("'$selection'"=="selection3"){print 500}else if ("'$selection'"=="selection4"){print 500}else if ("'$selection'"=="selection5"){print 250}}'`
space=`echo " " | awk '{if ("'$selection'"=="selection1"){print 10} else if ("'$selection'"=="selection2"){print 10} else if ("'$selection'"=="selection3"){print 4}else if ("'$selection'"=="selection4"){print 4}else if ("'$selection'"=="selection5"){print 2}}'`
echo "python patch2tfrecords.py " $dirtrainin " " $patchdir " " $selection " " $rayonmin " " $rayonmax " " $patchsize " " $space " " $enhanced " " $tfrecordtrain " " $dirtestin " " $tfrecordtest > scriptpatch.bat
echo "python convolutional_av_withqueue.py " $tfrecordtrain " " $tfrecordtest " " $checkpointdir " " $summarydir " " $numepoch > scripttraining.bat
echo "scriptpatch.bat" > scriptall.bat
echo "scripttraining.bat" >> scriptall.bat
# cmd /C "scriptpatch.bat"
# cmd /C "scripttraining.bat"