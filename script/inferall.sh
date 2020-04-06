dirout=E:/Fantin/AngioData/outputs
dirin=E:/Fantin/AngioData/TEST
gt=E:/Fantin/AngioData/TEST/seg

mkdir E:/Fantin/AngioData/checkpoints
mkdir E:/Fantin/AngioData/checkpoints/train70
mkdir E:/Fantin/AngioData/TRAIN_down/patch

#../../script/trainbat_var.sh median selection1
../../script/trainbat_var.sh tophat selection1
scriptpatch.bat

scripttraining.bat


../../script/trainbat_var.sh image selection1

../../script/trainbat_var.sh median selection2
../../script/trainbat_var.sh median selection3 FantinNetAngio_upscale
../../script/trainbat_var.sh median selection3 FantinNetAngio_upscale_unet
../../script/trainbat_var.sh median selection3 FantinNetAngio_upscale_unet_moredepth
../../script/trainbat_var.sh image selection3 FantinNetAngio_upscale_unet

../../script/trainbat_var.sh imageback selection3 FantinNetAngio_upscale_unet
# avec le catheter dans le background
../../script/trainbat_var.sh imageback selection3 FantinNetAngio_upscale_unet catheter


../../script/trainbat_var.sh image selection4 FantinNetAngio_upscale_unet_moredepth

../../script/trainbat_var.sh imageback selection3 FantinNetAngio_upscale_unet augm

../../script/trainbat_var.sh imageback selection3 FantinNetAngio_upscale_unet reinf

../../script/trainbat_var.sh image selection3 FantinNetAngio_upscale_unet reinf

../../script/trainbat_var_down.sh imageback selection5 FantinNetAngio_upscale_unet reinf


../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection1_train70 $gt catinvess gray selection1 1
../../script/create_bat_var.sh $dirin tophat upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_tophat_upscale_gray_selection1_train70 $gt catinvess gray selection1 1
../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_image_upscale_gray_selection1_train70 $gt catinvess gray selection1 1
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection2_train70 $gt catinvess gray selection2 1

../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection1_post_train70 $gt catinvess gray selection1 1
../../script/create_bat_var.sh $dirin tophat upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_tophat_upscale_gray_selection1_train70 $gt catinvess gray selection1 1
../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_image_upscale_gray_selection1_train70 $gt catinvess gray selection1 1
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection2_train70 $gt catinvess gray selection2 1 0
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection3_train70 $gt catinvess gray selection3 1 0
#GOOD
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_median_upscale_gray_selection3_unet_train70 $gt catinvess gray selection3 1 0
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_median_upscale_gray_selection3_unet_post_train70 $gt catinvess gray selection3 1 500

../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale_unet_moredepth inference_image_enh.py $dirout/output_median_upscale_gray_selection3_unet_moredepth_train70 $gt catinvess gray selection3 1 0
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale_unet_moredepth inference_image_enh.py $dirout/output_median_upscale_gray_selection3_unet_moredepth_post_train70 $gt catinvess gray selection3 1 500

../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_image_upscale_gray_selection3_unet_train70 $gt catinvess gray selection3 1 0
../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_image_upscale_gray_selection3_unet_post_train70 $gt catinvess gray selection3 1 500

../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_train70 $gt catinvess gray selection3 0 0
../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_post_train70 $gt catinvess gray selection3 0 500


../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet_moredepth inference_image_enh.py $dirout/output_image_upscale_gray_selection4_unet_train70 $gt catinvess gray selection4 1 0
../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet_moredepth inference_image_enh.py $dirout/output_image_upscale_gray_selection4_unet_post_train70 $gt catinvess gray selection4 1 500

../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_augm_train70 $gt catinvess gray selection3 0 0 augm


../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf0_train70 $gt catinvess gray selection3 0 0 0 reinf
../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf0_post_train70 $gt catinvess gray selection3 0 500 0 reinf
../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf0_posthyst_train70 $gt catinvess gray selection3 0 4000 30 reinf

../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_image_upscale_gray_selection3_unet_reinf0_train70 $gt catinvess gray selection3 0 0 0 reinf
../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_image_upscale_gray_selection3_unet_reinf0_post_train70 $gt catinvess gray selection3 0 1000 0 reinf

../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf1_train70 $gt catinvess gray selection3 0 0 0 reinf
../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf1_post_train70 $gt catinvess gray selection3 0 1000 0 reinf
../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf1_post2_train70 $gt catinvess gray selection3 0 1000 0 reinf 36
../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf1_posthyst_train70 $gt catinvess gray selection3 0 4000 30 reinf

../../script/create_bat_var.sh $dirin imageback upscale train70 FantinNetAngio_upscale_unet inference_image_enh_3c.py $dirout/output_imageback_upscale_gray_selection3_unet_reinf1_train70_resol05 $gt catinvess gray selection3 0 0 0 reinf 100 0.5


../../script/create_bat_var.sh $dirin image upscale train70 FantinNetAngio_upscale_unet inference_image_enh.py $dirout/output_image_upscale_gray_selection3_unet_reinfdbg_train70 $gt catinvess gray selection3 0 0 0 reinf



../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection2_post_train70 $gt catinvess gray selection2 1 500
../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection3_post_train70 $gt catinvess gray selection3 1 500



../../script/create_bat_var.sh $dirin image deconv train10 FantinNetAngio_deconv.py inference_image_ori.py $dirout/output_image_deconv_train10 $gt catinvess color selection1 0

../../script/create_bat_var.sh $dirin median deconv train10 FantinNetAngio_deconv.py inference_image_enh.py $dirout/output_median_deconv_train10 $gt catinvess color selection1 0

../../script/create_bat_var.sh $dirin median upscale train10 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_train10 $gt catinvess color selection1 0

../../script/create_bat_var.sh $dirin tophat upscale train10 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_tophat_upscale_train10 $gt catinvess color selection1 0

../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh_3c.py $dirout/output_median_upscale_train70 $gt catinvess color selection1 0

../../script/create_bat_var.sh $dirin median upscale train70 FantinNetAngio_upscale.py inference_image_enh.py $dirout/output_median_upscale_gray_selection1_train70 $gt catinvess gray selection1 1


cmd /C "python getROCcurve.py combine "$dirout