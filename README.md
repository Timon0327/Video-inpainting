# Video-inpainting

# Related work
- Survery: Ilan, S., Shamir, A.: A survey on data-driven video completion. In: Computer Graphics Forum. vol. 34, pp. 60–85 (2015)

1. Patch-based synthesis (Limitation: slow and they can only remix existing patches in the video, cannot handle non-repetitive structures such as faces)
   - Wexler, Y., Shechtman, E., Irani, M.: Space-time completion of video. TPAMI (3), 463–476 (2007)
   - Newson, A., Almansa, A., Fradet, M., Gousseau, Y., Pérez, P.: Video inpainting of complex scenes. SIAM Journal on Imaging Sciences (2014)
   - Huang, J.B., Kang, S.B., Ahuja, N., Kopf, J.: Temporally coherent completion of dynamic video. ACM Transactions on Graphics (TOG) (2016)

2. Learning-based synthesis
   
   - Deep neural network:
      - Wang, C., Huang, H., Han, X., Wang, J.: Video inpainting by jointly learning temporal structure and spatial details. In: AAAI (2019):
         - first work to use deep neural networks
         - 3D CNN fir temporal structure prediction, 2D CNN for spatial detail recovering
      - ** Lee, S., Oh, S.W., Won, D., Kim, S.J.: Copy-and-paste networks for deep video inpainting. In: ICCV (2019):
         - DNN-based 
         - Align the frames
   
   - Gan:
      -  J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang. Generative image inpainting with contextual attention. In: CVPR (2018) (image inpainting):
         - Gan
         - Attention
      -  Ya-Liang Chang, Zhe Yu Liu, and Winston Hsu. Vornet: Spatio-temporally consistent video inpainting for object removal. In: CVPR (2019):
         - FlowNet2 pre-trained to calculated the raw optial flow
         - Gan from Yu
         - Convolustional LSTM
         - **TempoGan Loss
      - Chang, Y.L., Liu, Z.Y., Hsu, W.: Free-form video inpainting with 3D gated convolution and temporal patchgan. In: ICCV (2019):
         - **Temporal PatchGAN Loss 
         - Generator network
         - 3D gated convolution 
      - Sagong, Min-cheol, et al. "Pepsi: Fast image inpainting with parallel decoding network." In: CVPR (2019):
         - Based on CAM
         - Coarse-to-fine network: coarse network and refinement network
   - Optimal Flow:
      - Huang, J.B., Kang, S.B., Ahuja, N., Kopf, J.: Temporally coherent completion of dynamic video. ACM Transactions on Graphics (TOG) (2016) (state-of-the-art):
         - Forward and backward optimal flow
      - Xu, R., Li, X., Zhou, B., Loy, C.C.: Deep flow-guided video inpainting. In: CVPR (2019):
         - Using FlowNet2
         - ResNet50
      -  Ya-Liang Chang, Zhe Yu Liu, and Winston Hsu. Vornet: Spatio-temporally consistent video inpainting for object removal. In: CVPR (2019):
         - FlowNet2 pre-trained to calculated the raw optial flow
         - Gan from Yu
         - Convolustional LSTM
         - TempoGan Loss
      - Gao C, Saraf A, Huang J B, et al. Flow-edge guided video completion. In: ECCV (2020):
         - Flow egdge
            - FlowNet2
            - Canny edge detector
            - EdgeConne
         - Non-local flow
         - Seamless blending
         - TempoGan
      - Kim, D., Woo, S., Lee, J.Y., Kweon, I.S.: Deep video inpainting. In: CVPR (2019):
         - Optimal Flow
         - ConvLSTM
         - Flow loss and wrap loss
         - Generated frame and generated inpainting frame
            
         
   - Attention:
      -  J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang. Generative image inpainting with contextual attention. In: CVPR (2018):
         - Gan
         - Attention
         - Coarse-to-fine network: coarse network and refinement network
      - Oh, S.W., Lee, S., Lee, J.Y., Kim, S.J.: Onion-peel networks for deep video completion. In: ICCV (2019):
         - Asymmetric Attention Block
      - Sagong, Min-cheol, et al. "Pepsi: Fast image inpainting with parallel decoding network." In: CVPR (2019):
         - Based on CAM
         - Coarse-to-fine network: coarse network and refinement network
      - Zeng Y, Fu J, Chao H, et al. Learning pyramid-context encoder network for high-quality image inpainting. CVPR (2019):
         - High level feature to low level by Attention
   
   
   
   - Reinforcment Learning:
      - Han X, Zhang Z, Du D, et al. Deep reinforcement learning of volume-guided progressive view inpainting for 3d point scene completion from a single depth image. In: CVPR (2019):
         - volume guidance
         - 3D scene volume reconstruction
         - 2D depth map inpainting
         - Deep Q-network to choose the best view for large hole completion
         
  
  
   - Graph:
     - INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS. In ICLR (2020):
         - temporal graph attention
     - Hyper-SAGNN: a self-attention based graph neural network for hypergraphs. In ICLR (2020):
         - **hypergraph 
         - self-attention
     - (David, Google) Learning to Execute Programs with Instruction Pointer Attention Graph Neural Networks. In Nips (2020):
         - attention graph
     - Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting. In Nips (2020)



   
   
