[app]
title = FaceApp
package.name = faceapp
package.domain = org.faceapp
source.dir = .
source.include_exts = py,png,jpg,mp3,html,json,xml
version = 1.0
requirements = kivy, flask, opencv-python, numpy, requests, jinja2
orientation = portrait
android.permissions = INTERNET, CAMERA
android.api = 33
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
