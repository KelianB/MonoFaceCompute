echo "Pulling submodules"

# for repo in INFERNO face-parsing.PyTorch MODNet SMIRK DSINE omnidata
for repo in INFERNO face-parsing.PyTorch MODNet SMIRK DSINE sapiens IntrinsicAnything
do
    echo $repo
    # Pull non-recursively
    git submodule update --init submodules/$repo &&
    # Apply the patch if it exists
    if [ -f submodules/$repo.patch ]; then
        (cd submodules/$repo && patch -p1 <../$repo.patch)
    fi
    # Pull recursively
    git submodule update --recursive submodules/$repo
done