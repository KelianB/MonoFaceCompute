echo "Pulling submodules"

# for repo in INFERNO face-parsing.PyTorch MODNet SMIRK DSINE omnidata
for repo in INFERNO face-parsing.PyTorch MODNet SMIRK DSINE sapiens
do
    echo $repo
    # Pull non-recursively
    git submodule update --init submodules/$repo &&
    # Apply the patch
    cd submodules/$repo
    patch -p1 <../$repo.patch
    cd ../../
    # Pull recursively
    git submodule update --recursive submodules/$repo
done