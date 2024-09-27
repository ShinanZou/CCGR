#CCGR

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_PAR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_PAR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_PAR.yaml --phase test --log_to_file &&

#python CCGR_EVA.py --model Baseline --savename GaitBase_SIL &&
#python CCGR_EVA.py --typedata par --model Baseline --savename GaitBase_PAR &&
#python CCGR_EVA.py --model GaitSet --savename GaitSet_CCGR_SIL --dist_model 2 &&
#python CCGR_EVA.py --typedata par --model GaitSet --savename GaitSet_CCGR_PAR --dist_model 2 &&
#python CCGR_EVA.py --model DeepGaitV2 --savename DeepGaitV2_SIL &&
#python CCGR_EVA.py --typedata par --model DeepGaitV2 --savename DeepGaitV2_PAR

#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_PAR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR.yaml --phase test --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_PAR.yaml --phase test --log_to_file &&

#python CCGR_EVA.py --model GaitPart --savename GaitPart_CCGR_SIL &&
#python CCGR_EVA.py --typedata par --model GaitPart --savename GaitPart_CCGR_PAR &&
#python CCGR_EVA.py --model GaitGL --savename GaitGL_SIL --dist_model 2 &&
#python CCGR_EVA.py --typedata par --model GaitGL --savename GaitGL_PAR --dist_model 2



#CCGR-MINI
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_MINI.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_MINI_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_MINI.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_MINI_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_MINI.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_MINI_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_PAR_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_MINI_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_MINI_PAR_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_MINI.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_MINI_PAR.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_MINI.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_MINI_PAR.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_MINI.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_MINI_PAR.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_DA.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_PAR_DA.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_MINI_DA.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_MINI_PAR_DA.yaml --phase test --log_to_file &&