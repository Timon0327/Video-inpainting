# Video-inpainting

# Related work
1. Patch-based synthesis (Limitation: slow and they can only remix existing patches in the video, cannot handle non-repetitive structures such as faces)
   - Wexler, Y., Shechtman, E., Irani, M.: Space-time completion of video. TPAMI (3), 463–476 (2007)
   - Newson, A., Almansa, A., Fradet, M., Gousseau, Y., Pérez, P.: Video inpainting of complex scenes. SIAM Journal on Imaging Sciences (2014)
   - Huang, J.B., Kang, S.B., Ahuja, N., Kopf, J.: Temporally coherent completion of dynamic video. ACM Transactions on Graphics (TOG) (2016)
2. Learning-based synthesis
   - Deep neural network:
      - Wang, C., Huang, H., Han, X., Wang, J.: Video inpainting by jointly learning temporal structure and spatial details. In: AAAI (2019):
         - first work to use deep neural networks
         - 3D CNN fir temporal structure prediction, 2D CNN for spatial detail recovering
   - Gan:
      -  Ya-Liang Chang, Zhe Yu Liu, and Winston Hsu. Vornet: Spatio-temporally consistent video inpainting for object removal. In: CVPR (2019):
         - FlowNet2 pre-trained to calculated the raw optial flow
         - Gan from Yu
         - Convolustional LSTM
         - TempoGan Loss
      - Chang, Y.L., Liu, Z.Y., Hsu, W.: Free-form video inpainting with 3D gated convolution and temporal patchgan. In: ICCV (2019):
         - Temporal PatchGAN
         - 3D gated convolution 
   - Wang, C., Huang, H., Han, X., Wang, J.: Video inpainting by jointly learning temporal structure and spatial details. In: AAAI (2019):
      - first work to use deep neural networks
      - 3D CNN fir temporal structure prediction, 2D CNN for spatial detail recovering
   
   
