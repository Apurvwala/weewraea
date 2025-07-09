[app]
title = FaceApp
package.name = faceapp
package.domain = org.faceapp
source.dir = .
source.include_exts = py,png,jpg,mp3,html,json
version = 1.0
requirements = kivy==2.2.1,flask,opencv-contrib-python,numpy,requests,jinja2
orientation = portrait
android.permissions = INTERNET, CAMERA
android.api = 33
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True
android.archs = armeabi-v7a arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
