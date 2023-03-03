# ImageEditing
Editing Baselines


## For textual inversion

Library Version:
- Pytorch: 1.13.1+cu117
- diffusers: 0.13.1

```
cd textual_inversion
accelerate run --files [YOUR_DIR] --special_token [TOKEN] --initialize [WORD]
accelrate run --files dogs/ --special_token [dog] --initialize dog
```

## For Null-Text Inversion

Library Version:
- Pytorch: 1.13.1+cu117
- diffusers: 0.8.0

```
cd prompt-to-prompt
```
open null_text_w_ptp.ipynb, follow the instruction to do image editing
