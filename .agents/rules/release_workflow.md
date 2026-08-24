# Release & Code Push Workflow Rule

Whenever a version bump, release, or code update is requested to be pushed:
1. **Version & Documentation Sync**:
   - Update `VERSION` in `gui_app.py`
   - Update version in `README.md`
   - Add release notes in `CHANGELOG.md`
2. **Automated Verification**:
   - Run unit test suite (`python -m unittest discover -s . -p "test_*.py"`)
3. **Executable Build**:
   - Build standalone single executable (`pyinstaller --clean --noconfirm TeraboxClicker.spec`) to generate `dist/TeraboxClicker.exe`
4. **Git Commit & Tag Push**:
   - Commit changes and tag with semantic version (e.g. `v0.3.x`)
   - Push commit to `main` and tag to remote
5. **GitHub Release & Executable Upload**:
   - Ensure GitHub Release for the tag is created and `dist/TeraboxClicker.exe` is uploaded as the release asset
