#python train.py --log_dir ./log/test_ --fold 0 --save ./work/test_/certainty_loss_g2_save --is_uncertain --uncertain_weight 0.01 --gamma 2 --is_save_uncertain

#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/certainty_loss_g2_w0.0001 --is_uncertain --uncertain_weight 0.0001
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/certainty_loss_g2_w0.001 --is_uncertain --uncertain_weight 0.001
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/certainty_loss_g2_w0.1 --is_uncertain
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/certainty_loss_g2 --is_uncertain
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/certainty_loss_g1 --gamma 1 --is_uncertain
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/certainty_loss_g0.5 --gamma 0.5 --is_uncertain

#python train.py --log_dir ./log/test_ --fold 0 --save ./work/test_/test_model

#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/boundary_stable_1e-3 --loss boundary --boundary_method stable --boundary_alpha 0.001
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/boundary_stable_1e-4 --loss boundary --boundary_method stable --boundary_alpha 0.0001
#python train.py --log_dir ./log/losses --fold 0 --save ./work/losses/boundary_rump_0.005 --loss boundary --boundary_method rump


#python train.py --arch resunet3d --log_dir ./log/res_unet_3d --fold 0 --save ./work/res_unet_3d/res_unet_3d_fold_0
#python train.py --arch unet3d --log_dir ./log/unet_3d_fold_0 --fold 0 --save ./work/unet_3d/unet_3d_fold_0    
#python train.py --arch vnet --log_dir ./log/vnet --fold 0 --save ./work/vnet/vnet_fold_0    


#python train.py --arch unet3d --log_dir ./log/res_unet_3d --fold 1 --save ./work/res_unet_3d/res_unet_3d_fold_1    
#python train.py --arch unet3d --log_dir ./log/res_unet_3d --fold 2 --save ./work/res_unet_3d/res_unet_3d_fold_2    
#python train.py --arch unet3d --log_dir ./log/res_unet_3d --fold 3 --save ./work/res_unet_3d/res_unet_3d_fold_3    
#python train.py --arch unet3d --log_dir ./log/res_unet_3d --fold 4 --save ./work/res_unet_3d/res_unet_3d_fold_4    
#python train.py --arch unet3d --log_dir ./log/res_unet_3d --fold 5 --save ./work/res_unet_3d/res_unet_3d_fold_5    

#python train.py --arch unet3d --log_dir ./log/unet_3d --fold 1 --save ./work/unet_3d/unet3d_fold_1    
#python train.py --arch unet3d --log_dir ./log/unet_3d --fold 2 --save ./work/unet_3d/unet3d_fold_2    
#python train.py --arch unet3d --log_dir ./log/unet_3d --fold 3 --save ./work/unet_3d/unet3d_fold_3    
#python train.py --arch unet3d --log_dir ./log/unet_3d --fold 4 --save ./work/unet_3d/unet3d_fold_4    
#python train.py --arch unet3d --log_dir ./log/unet_3d --fold 5 --save ./work/unet_3d/unet3d_fold_5    

#python train.py --arch vnet --log_dir ./log/vnet --fold 1 --save ./work/vnet/vnet_fold_1    
#python train.py --arch vnet --log_dir ./log/vnet --fold 2 --save ./work/vnet/vnet_fold_2    
#python train.py --arch vnet --log_dir ./log/vnet --fold 3 --save ./work/vnet/vnet_fold_3    
#python train.py --arch vnet --log_dir ./log/vnet --fold 4 --save ./work/vnet/vnet_fold_4    
#python train.py --arch vnet --log_dir ./log/vnet --fold 5 --save ./work/vnet/vnet_fold_5    

python train.py --arch denseunet --log_dir ./log/cv --fold 1 --save ./work/cv/dice_first2_123_fold_1 --gpu 0
python train.py --arch denseunet --log_dir ./log/cv --fold 2 --save ./work/cv/dice_first2_123_fold_2 --gpu 0
python train.py --arch denseunet --log_dir ./log/cv --fold 3 --save ./work/cv/dice_first2_123_fold_3 --gpu 0
python train.py --arch denseunet --log_dir ./log/cv --fold 4 --save ./work/cv/dice_first2_123_fold_4 --gpu 0
python train.py --arch denseunet --log_dir ./log/cv --fold 5 --save ./work/cv/dice_first2_123_fold_5 --gpu 0
