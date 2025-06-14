from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # Directory where datasets are stored
_C.imb_factor = None  # for long-tailed cifar dataset
_C.head_init_folder = None  # path to class mean file

_C.backbone = ""
_C.resolution = 224

# Directory to save the output files (like log.txt and model weights)
_C.output_dir = None
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None  # use manual seed
_C.deterministic = False  # output deterministic results
_C.num_workers = 20
_C.prec = "fp16"  # fp16, fp32, amp

_C.num_epochs = 10
_C.batch_size = 128
# for gradient accumulation, must be a divisor of batch size
_C.micro_batch_size = 128
_C.lr = 0.01
_C.scheduler = "CosineAnnealingLR"
_C.weight_decay = 5e-4
_C.momentum = 0.9
# "CE" / "Focal" / "LDAM" / "CB" / "GRW" / "BS" / "LA" / "LADE"
_C.loss_type = "LA"

_C.classifier = "CosineClassifier"
_C.scale = 25  # for cosine classifier

_C.fine_tuning = False  # end-to-end fine-tuning

_C.full_tuning = False  # full fine-tuning
_C.bias_tuning = False  # only fine-tuning the bias
_C.ln_tuning = False  # only fine-tuning the layer norm
_C.bn_tuning = False  # only fine-tuning the batch norm (only for resnet)
_C.vpt_shallow = False
_C.vpt_deep = False
_C.adapter = False
_C.adaptformer = False
_C.lora = False
_C.lora_mlp = False
_C.ssf_attn = False
_C.ssf_mlp = False
_C.ssf_ln = False
_C.mask = False  # fine-tuning a specific proportion of all parameters
_C.partial = None  # fine-tuning (or parameter-efficient fine-tuning) partial block layers
_C.vpt_len = None  # length of VPT sequence
_C.adapter_dim = None  # bottle dimension for adapter / adaptformer / lora.
_C.adaptformer_scale = "learnable"  # "learnable" or scalar
_C.mask_ratio = None
_C.mask_seed = None
_C.flora = False

_C.init_head = None  # "text_feat" (only for CLIP) / "class_mean" / "1_shot" / "10_shot" / "100_shot" / "linear_probe"
_C.prompt = "default"  # "classname" / "default" / "ensemble" / "descriptor"
_C.tte = False  # test-time ensemble
_C.expand = 24  # expand the width and height of images for test-time ensemble
_C.tte_mode = "fivecrop"  # "fivecrop" / "tencrop" / "randaug"
_C.randaug_times = 1

_C.zero_shot = False  # zero-shot CLIP (only for CLIP)
_C.test_only = False  # load model and test
_C.test_train = False  # load model and test on the training set
_C.model_dir = None


# FLoRA CONFIGS
_C.use_flora = False

# Add the flora configuration node
_C.flora = CN()
_C.flora.default_rank = 4
_C.flora.default_lr = 1e-2

# Define the modules to apply flora
_C.flora.modules = ["q", "v", "k", "out", "mlp1", "mlp2"]

# Apply flora to all layers (indices 0 to 11)
_C.flora.apply_to_layers = list(range(12))  # Layers 0 to 11

# Initialize the layers list
_C.flora.layers = []

# Create default configurations for each layer
for idx in _C.flora.apply_to_layers:
    layer_cfg = CN()
    layer_cfg.index = idx
    layer_cfg.modules = CN()
    for module_name in _C.flora.modules:
        module_cfg = CN()
        module_cfg.rank = _C.flora.default_rank
        module_cfg.lr = _C.flora.default_lr
        # Assign the module configuration to the layer's modules
        setattr(layer_cfg.modules, module_name, module_cfg)
    # Append the layer configuration to the layers list
    _C.flora.layers.append(layer_cfg)
