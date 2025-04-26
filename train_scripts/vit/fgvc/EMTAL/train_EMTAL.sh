device=0
train_list=(1 0 0 0 0)

exp='train_EMTAL.py'
save_path='EMTAL-fgvc'

vt_name='vision_transformer_LoRA_drop_fade-weight_rank'
tm='LoRA'

reg_type=0 #######
center=0
vpt='None'
tao=10
beta=0 ########
drop_head=0
lr_scale=1.0
drop_path=0
batch_size=32
Lora_rank=4
warmup=5e-6
minlr=1e-5
dropout=0

echo "tuning-mode  ${tm} "
for aug in 2 ;do 
	for seed in 42;do  
		for lr_ in 0.001 0.0005; do 
			if [ ${train_list[0]} == 1 ]
			then
				echo "start training ${device} lr_ ${lr_} ${drop_head} ${beta} using  ssf ckpt"
				CUDA_VISIBLE_DEVICES=${device}  python  -m torch.distributed.launch --nproc_per_node=1  --master_port=`expr 14350 + ${device}`  \
					${exp} /data/datasets/FGVC --dataset cub2011 --num-classes 200 --simple-aug --model vit_base_patch16_224_in21k_ssf  \
					--batch-size ${batch_size} --epochs 100 \
					--opt adamw  --weight-decay 5e-2 \
					--warmup-lr ${warmup} --warmup-epochs 10  \
					--lr ${lr_} --min-lr ${minlr} --seed $seed \
					--drop-path ${drop_path} --img-size 224 \
					--model-ema --model-ema-decay 0.9998 \
					--output  output/${save_path} --reg-type ${reg_type} \
					--amp --tuning-mode ${tm} --pretrained     --center ${center}\
					--tao ${tao}   --beta ${beta} --vt-name ${vt_name}  --drop-head ${drop_head} --no-prefetcher \
					--Lora_alpha ${Lora_rank} --mlp_r 4 --attn_cls 0 --ckpt_kmeans '/root/autodl-tmp/workspace-0420/EMTAL-dev/16e_weight_basic_wo-grad.pth' --aug_epoch ${aug} --root_dir '/root/autodl-tmp/EMTAL' --mtl_type '2sample'
				sleep 5s
				# device=`expr ${device} + 1`
			fi

		done
	done
done