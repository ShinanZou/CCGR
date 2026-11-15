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
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_MINI.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_MINI_PAR.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_DA.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_PAR_DA.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_MINI_DA.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_MINI_PAR_DA.yaml --phase test --log_to_file &&



##CCGR-3D
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_3D_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_3D_PAR.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_3D_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_3D_PAR_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_3D_DA.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_3D_PAR_DA.yaml --phase train --log_to_file &&
###sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_3D_PAR.yaml --phase train --log_to_file &&
##sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/smplgait/smplgait_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/skeletongait/skeletongait_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/skeletongait/skeletongait++_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/biggait/biggait_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 1800 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_with_3dgait_CCGR_3D_DA.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_with_3dgait_CCGR_3D.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_with_3dgait_CCGR_3D.yaml --phase train --log_to_file

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/svgait_CCGR_3D_DA.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgait_CCGR_3D_DA.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/vrn_CCGR_3D_DA_p.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_1.5.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.yaml --phase train --log_to_file &&
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2one.yaml --phase train --log_to_file
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5_2.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5s.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5_ceshi.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.3.yaml --phase train --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.0.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_1.5.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_1.0.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_3.0.yaml --phase train --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5ss.yaml --phase train --log_to_file
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_3.yaml --phase train --log_to_file
#sleep 600 &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_CCGR_Mini.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gpgait/gpgait_CCGR_3D.yaml --phase train --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gpgait/gpgait_CCGR_Mini.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/hstl/hstl_CCGR_3D.yaml --phase train --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5_ceshiwucha.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5_ceshiwucha2.yaml --phase test --log_to_file

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitset/gaitset_CCGR_3D_PAR.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitpart/gaitpart_CCGR_3D_PAR.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_3D_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_3D_PAR_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_3D_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_CCGR_3D_PAR_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitgl/gaitgl_CCGR_3D_PAR.yaml --phase test --log_to_file   &&
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/smplgait/smplgait_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/skeletongait/skeletongait_CCGR_3D.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/skeletongait/skeletongait++_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/3dgait/svgait_CCGR_3D_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/biggait/biggait_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/deepgait/deepgait_with_3dgait_CCGR_3D_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50006 opengait/main.py --cfgs ./configs/gpgait/gpgait_CCGR_3D.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/svgait_CCGR_3D_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgait_CCGR_3D_DA.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_1.5.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_2.5_1000.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50007 opengait/main.py --cfgs ./configs/3dgait/3dgaitpp_CCGR_3D_DA_3.yaml --phase test --log_to_file


#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_addr=127.0.0.1 --master_port=50003 opengait/main.py --cfgs ./configs/MSAFF/MSAFF_CCGR_3D.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_addr=127.0.0.1 --master_port=50003 opengait/main.py --cfgs ./configs/MSAFF/MSAFF_CCGR_3D_experiment.yaml --phase train --log_to_file &&
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_addr=127.0.0.1 --master_port=50003 opengait/main.py --cfgs ./configs/MSAFF/MSAFF_CCGR_MINI.yaml --phase train --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_addr=127.0.0.1 --master_port=50003 opengait/main.py --cfgs ./configs/MSAFF/MSAFF_CCGR_3D.yaml --phase test --log_to_file &&
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  --master_addr=127.0.0.1 --master_port=50003 opengait/main.py --cfgs ./configs/MSAFF/MSAFF_CCGR_3D_experiment.yaml --phase test --log_to_file
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_addr=127.0.0.1 --master_port=50003 opengait/main.py --cfgs ./configs/MSAFF/MSAFF_CCGR_MINI.yaml --phase test --log_to_file
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=50005 opengait/main.py --cfgs ./configs/gaitbase/gaitbase_CCGR_MINI_DA_deep.yaml --phase train --log_to_file