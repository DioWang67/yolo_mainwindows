# from PyInstaller.utils.hooks import collect_submodules

# hiddenimports = collect_submodules('tensorflow')


from PyInstaller.utils.hooks import collect_dynamic_libs

hiddenimports = ['tensorflow']
binaries = collect_dynamic_libs('tensorflow')
