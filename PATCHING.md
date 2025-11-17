# Updating patches for submodules

This is a memo for updating the submodules patch files.

- Make changes within individual submodules.
- Stage those changes temporarily but **do not** commit.
- Run these commands:

```bash
(cd ./submodules/face-parsing.PyTorch && git --no-pager diff --no-color --staged > ../face-parsing.PyTorch.patch)
(cd ./submodules/MODNet && git --no-pager diff --no-color --staged > ../MODNet.patch)
(cd ./submodules/SMIRK && git --no-pager diff --no-color --staged > ../SMIRK.patch)
(cd ./submodules/INFERNO && git --no-pager diff --no-color --staged > ../INFERNO.patch)
(cd ./submodules/DSINE && git --no-pager diff --no-color --staged > ../DSINE.patch)
(cd ./submodules/omnidata && git --no-pager diff --no-color --staged > ../omnidata.patch)
(cd ./submodules/sapiens && git --no-pager diff --no-color --staged > ../sapiens.patch)
```
 