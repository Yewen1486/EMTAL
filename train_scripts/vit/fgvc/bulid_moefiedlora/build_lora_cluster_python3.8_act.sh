device=0 # 5
train_list=(1 1 1 1 1)
train_list=(1 0 0 0 0)
# echo "count down 2.5"
# sleep 2.1h

exp='Get_clusters.py'
# exp='train_LoRA_union4_eval3.8_grad_orth.py'
# exp='train_LP_union4_FPT.py'
save_path='LoRA_union_Lrscheduler_disolve/test'

# basic 
lr_cub=0.005
lr_dogs=0.0001
lr_birds=0.0001 # 2e-4
lr_cars=0.01 #
lr_flowers=0.005

vt_name='vision_transformer_LoRA_drop_fade-weight_rank'
tm='cluster'
# tm='linear_probe'

reg_type=0 #######
center=0
vpt='None'
tao=10
beta=0 ########
drop_head=0
lr_scale=1.0
drop_path=0
batch_size=32
Lora_alpha=1

# echo "sleep 10h"
# sleep 10h
# 5e-3 * 8
echo "tuning-mode  ${tm} "
for dropout in 0; do #; do #; do # 
	# for lr_ in 1e-4 5e-4 1e-3 5e-3 1e-2 ; do # 0.1 0.3 
	for lr_ in 0.00025; do # 减小为一半x
        if [ ${train_list[0]} == 1 ]
        then
			# lr_=$(echo "${lr_cub} * ${lr_scale}" | bc -l)
			# CUDA_VISIBLE_DEVICES=${device}  torchrun --nproc_per_node=1  --master_port=`expr 14650 + ${device}`  

			echo "start training ${device} lr_ ${lr_} ${drop_head} ${beta} using  ssf ckpt"
			CUDA_VISIBLE_DEVICES=${device}  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=`expr 14650 + ${device}` \
				${exp} /data/datasets/FGVC --dataset cub2011 --num-classes 200 --simple-aug --model vit_base_patch16_224_in21k_ssf  \
				--batch-size ${batch_size} --epochs 100 \
				--opt adamw  --weight-decay 5e-2 \
				--warmup-lr 5e-6 --warmup-epochs 10  \
				--lr ${lr_} --min-lr 1e-6 \
				--drop-path ${drop_path} --img-size 224 \
				--model-ema --model-ema-decay 0.9998 \
				--output  output_beta/${save_path}/cub2011 --reg-type ${reg_type} \
				--amp --tuning-mode ${tm} --pretrained     --center ${center} --ckpt_output '16e_weight_basic_wo-grad.pth' \
				--tao ${tao}   --beta ${beta} --vt-name ${vt_name}  --drop-head ${drop_head} --no-prefetcher #\
				# --Lora_alpha ${Lora_alpha}  --mlp_r 4 #--weighted-loss  #####
			sleep 1m
			# device=`expr ${device} + 1`
		fi

		
        if [ ${train_list[1]} == 1 ]
        then
		### 你设置了min lr 
			# sleep 1m
			# lr_=$(echo "${lr_cars} * ${lr_scale}" | bc -l)
			# echo "using drop = 0.1!!!!!"
			echo "start training ${device} lr_ ${lr_} ${drop_head} ${beta}"
			CUDA_VISIBLE_DEVICES=${device}  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=`expr 14650 + ${device}`  \
				${exp} /data/datasets/FGVC/stanford_cars/output --dataset stanford_cars --num-classes 196 --val-split val  --simple-aug --model vit_base_patch16_224_in21k  \
				--batch-size 32 --epochs 100 \
				--opt adamw  --weight-decay 0.05 \
				--warmup-lr 5e-8 --warmup-epochs 10  \
				--lr ${lr_} --min-lr 1e-4 \
				--drop-path ${drop_path} --img-size 224 \
				--model-ema --model-ema-decay 0.9998  \
				--output  output_beta/${save_path}/stanford_cars --reg-type ${reg_type} \
				--amp --tuning-mode ${tm} --pretrained    --center ${center} \
				--tao ${tao}   --beta ${beta} --vt-name ${vt_name}  --drop-head ${drop_head} 
			sleep 1m
			# echo "start training ${device} lr_ ${lr_} "
			# device=`expr ${device} + 1`
		fi

		
        if [ ${train_list[2]} == 1 ]
        then
			# lr_=$(echo "${lr_dogs} * ${lr_scale}" | bc -l)
			# echo "using drop = 0.1!!!!!"
			echo "start training ${device} lr_ ${lr_} ${drop_head} ${beta}"
			CUDA_VISIBLE_DEVICES=${device}  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=`expr 14650 + ${device}`  \
				${exp} /data/datasets/FGVC/stanford_dogs  --dataset stanford_dogs --num-classes 120 --simple-aug    --model vit_base_patch16_224_in21k  \
				--batch-size 32 --epochs 100 \
				--opt adamw  --weight-decay 0.05 \
				--warmup-lr 5e-8 --warmup-epochs 10  \
				--lr ${lr_} --min-lr 1e-6 \
				--drop-path ${drop_path} --img-size 224 \
				--model-ema --model-ema-decay 0.9998 \
				--output  output_beta/${save_path}/stanford_dogs --reg-type ${reg_type} \
				--amp --tuning-mode ${tm} --pretrained     --center ${center}\
				--tao ${tao}   --beta ${beta} --vt-name ${vt_name}  --drop-head ${drop_head} 
			sleep 1m
			# device=`expr ${device} + 1`
			# echo "start training ${device} lr ${lr_} "
		fi

		
        if [ ${train_list[3]} == 1 ]
        then
			# lr_=$(echo "${lr_birds} * ${lr_scale}" | bc -l)
			echo "start training ${device} lr_ ${lr_} ${drop_head} ${beta}"
			CUDA_VISIBLE_DEVICES=${device}  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=`expr 14610 + ${device}`  \
				${exp} /data/datasets/FGVC/NABirds --dataset nabirds --num-classes 555  --simple-aug --model vit_base_patch16_224_in21k  \
				--batch-size 32 --epochs 100 \
				--opt adamw  --weight-decay 0.05 \
				--warmup-lr 5e-8 --warmup-epochs 10  \
				--lr ${lr_} --min-lr 1e-6 \
				--drop-path ${drop_path} --img-size 224 \
				--model-ema --model-ema-decay 0.9998  \
				--output  output_beta/${save_path}/nabirds --reg-type ${reg_type} \
				--amp --tuning-mode ${tm} --pretrained    --center ${center}\
				--tao ${tao}   --beta ${beta} --vt-name ${vt_name}  --drop-head ${drop_head} --drop ${dropout} 
			sleep 1m
			# device=`expr ${device} + 1`
			# echo "start training ${device} lr_ ${lr_} "
		fi
		
        if [ ${train_list[4]} == 1 ]
		then
			# lr_=$(echo "${lr_flowers} * ${lr_scale}" | bc -l)
			echo "start training ${device} lr_ ${lr_} ${drop_head} ${beta}"
			CUDA_VISIBLE_DEVICES=${device}  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=`expr 14670 + ${device}`  \
				${exp} /data/datasets/FGVC/102flowers/output  --dataset oxford_flowers --num-classes 102 --val-split val --simple-aug --model vit_base_patch16_224_in21k  \
				--batch-size 32 --epochs 100 \
				--opt adamw  --weight-decay 0.05 \
				--warmup-lr 5e-8 --warmup-epochs 10  \
				--lr ${lr_} --min-lr 1e-6 \
				--drop-path ${drop_path} --img-size 224 \
				--model-ema --model-ema-decay 0.999  \
				--output  output_beta/${save_path}/oxford_flowers_final \
				--reg-type ${reg_type} \
				--amp --tuning-mode ${tm} --pretrained    --center ${center}\
				--tao ${tao}   --beta ${beta} --vt-name ${vt_name}  --drop-head ${drop_head} 
			# device=`expr ${device} + 1`
			sleep 1m
		fi

	done
done