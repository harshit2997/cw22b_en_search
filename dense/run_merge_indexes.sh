sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=150G --time=0 --nodelist=boston-2-25 \
  --wrap="python -m merge_indexes --inputs /ssd/hmehrotr/CW22_ind_en00_00_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_01_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_02_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_03_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_04_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_05_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_06_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_07_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_08_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_09_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_10_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_11_contrmarco_mean \
        --output /ssd/hmehrotr/CW22_ind_en00_00_11_contrmarco_mean "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=150G --time=0 --nodelist=boston-2-27 \
  --wrap="python -m merge_indexes --inputs /ssd/hmehrotr/CW22_ind_en00_12_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_13_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_14_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_15_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_16_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_17_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_18_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_19_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_20_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_21_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_22_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_23_contrmarco_mean \
        --output /ssd/hmehrotr/CW22_ind_en00_12_23_contrmarco_mean "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=150G --time=0 --nodelist=boston-2-31 \
  --wrap="python -m merge_indexes --inputs /ssd/hmehrotr/CW22_ind_en00_24_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_25_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_26_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_27_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_28_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_29_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_30_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_31_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_32_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_33_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_34_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_35_contrmarco_mean \
        --output /ssd/hmehrotr/CW22_ind_en00_24_35_contrmarco_mean "

sbatch -n 8 -N 1 -p gpu --gpus=1 --mem=150G --time=0 --nodelist=boston-2-29 \
  --wrap="python -m merge_indexes --inputs /ssd/hmehrotr/CW22_ind_en00_36_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_37_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_38_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_39_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_40_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_41_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_42_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_43_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_44_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_45_contrmarco_mean /ssd/hmehrotr/CW22_ind_en00_46_contrmarco_mean \
        --output /ssd/hmehrotr/CW22_ind_en00_36_46_contrmarco_mean "
