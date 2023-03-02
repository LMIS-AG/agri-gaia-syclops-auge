# syclops-auge
Augmentation Environment (AugE) for syclops.  
Can be implemented as postprocessor into the job description.

Example:

```bash
  Augmentation/Auge:
    - type: "augmentation"
      bg_classes: [1] # classes considered background
      target_classes: [2] # classes subjected to augmentation
      use_sd: False # optional
      id: auge
      sources: ["main_cam_rgb", "main_cam_instance", "main_cam_semantic", "main_cam_depth"]
```

![](meilenstein_6_12.gif)