// -----JS CODE-----
// HandTrackingController.js
// Version: 0.0.2
// Event: Initialized
// Description: The primary script that drives the hand tracking template.

// @input Asset.Texture handTexture { "label":"Texture" }
// @input float handSize = 0.5 {"label":"Size", "widget":"slider", "min":0.0, "max":1.0, "step":0.01}
// @input float handOffsetX = 0.0 {"label":"Offset X","widget":"slider", "min":-1.0, "max":1.0, "step":0.01}
// @input float HandOffsetY = 0.0 {"label":"Offset Y", "widget":"slider", "min":-1.0, "max":1.0, "step":0.01}
// @input float handRotation = 0.0 {"label":"Rotate","widget":"slider", "min":0.0, "max":360.0, "step":0.5}
// @input float handAlpha = 1.0 {"label":"Alpha","widget":"slider", "min":0.0, "max":1.0, "step":0.1}
// @input bool smoothFollowing
// @input float smoothingSpeed {"showIf": "smoothFollowing"}

// @ui {"widget":"separator"}

// @input bool advanced = false
// @input Component.ObjectTracking handTracking {"showIf": "advanced"}
// @input Component.Image handImage {"showIf": "advanced"}
// @input Component.ScreenTransform smoothFollower {"showIf": "advanced"}

var initialized = false;
var targetToFollow;

// Object Tracking callbacks needs to be run before anything else.
handTrackingCallBacks();

function onLensTurnOn() {
    if (initializeObjectTracking()) {
        configureHandTracking({
            texture: script.handTexture,
            objectsToTransform: script.handImage,
            scaleToAdd: script.handSize,
            xOffsetToAdd: script.handOffsetX,
            yOffsetToAdd: script.HandOffsetY,
            rotationToAdd: script.handRotation,
            imageAlpha: script.handAlpha
        });
    }
}

function onUpdate(eventData) {
    if (initialized && script.smoothFollowing) {
        smoothFollow(targetToFollow, script.smoothFollower, script.smoothingSpeed);
    }
}

function initializeObjectTracking() {
    if (!script.handTexture) {
        print("HandTrackingController, ERROR: Please assign a texture in the texture input.");
        return false;
    }

    if (!script.handTracking) {
        print("HandTrackingController, ERROR: Please assign Hand Tracking Object under the advanced checkbox.");
        return false;
    }

    if (!script.handImage) {
        print("HandTrackingController, ERROR: Please make sure Hand Image object exist and assign Hand Image object under the advanced checkbox.");
        return false;
    }

    if (!script.smoothFollower) {
        print("HandTrackingController, ERROR: Please make sure Smooth Follower object exist and assign the object under the advanced checkbox.");
        return false;
    }

    if (script.smoothFollowing) {
        script.handImage.getSceneObject().setParent(script.smoothFollower.getSceneObject());
        script.handImage.enabled = false;
    }

    targetToFollow = script.handTracking.getSceneObject().getComponent("Component.ScreenTransform");

    initialized = true;

    return true;
}

function handTrackingCallBacks() {
    if (script.handTracking) {
        script.handTracking.onObjectFound = wrapFunction(script.handTracking.onObjectFound, function() {
            script.handImage.enabled = true;

            var rootObjects = global.scene.getRootObjectsCount();
            for (var i = 0; i < rootObjects; i++) {
                var childObject = global.scene.getRootObject(i);
                sendObjectEvent(childObject, true);
            }
        });

        script.handTracking.onObjectLost = wrapFunction(script.handTracking.onObjectLost, function() {
            script.handImage.enabled = false;

            var rootObjects = global.scene.getRootObjectsCount();
            for (var i = 0; i < rootObjects; i++) {
                var childObject = global.scene.getRootObject(i);
                sendObjectEvent(childObject, false);
            }
        });
    }
}

function configureHandTracking(option) {
    var scaleMultiplier = 3;
    var offsetMultiplier = 3;
    var rotationDivider = 0.0175;

    option.objectsToTransform.mainPass.baseColor =
        setAlpha(option.objectsToTransform.mainPass.baseColor, option.imageAlpha);

    option.objectsToTransform.mainPass.baseTex = option.texture;

    var screenTransform = option.objectsToTransform.getSceneObject().getComponent("Component.ScreenTransform");

    var anchors = screenTransform.anchors;

    anchors.left = -option.scaleToAdd * scaleMultiplier;
    anchors.right = option.scaleToAdd * scaleMultiplier;

    anchors.top = option.scaleToAdd * scaleMultiplier;
    anchors.bottom = -option.scaleToAdd * scaleMultiplier;

    var offsetVec = new vec3(option.xOffsetToAdd, option.yOffsetToAdd, 0).uniformScale(offsetMultiplier);

    anchors.left += offsetVec.x;
    anchors.right += offsetVec.x;
    anchors.top += offsetVec.y;
    anchors.bottom += offsetVec.y;

    var rotateTo = screenTransform.rotation.multiply(
        quat.angleAxis((option.rotationToAdd * rotationDivider), vec3.forward())
    );

    screenTransform.rotation = rotateTo;
}

function smoothFollow(target, follower, smoothing) {
    var desiredRotation = target.rotation;
    var desiredPosition = target.anchors.getCenter();
    var desiredSize = target.anchors.getSize();

    var currentCenter = follower.anchors.getCenter();
    var currentSize = follower.anchors.getSize();

    // Calculate the speed to get from current to desired
    var t = Math.min(smoothing * getDeltaTime(), 1.0);
    var sizeT = Math.min(smoothing * 10 * getDeltaTime(), 1.0);

    // Calculate the next position
    var newCenter = vec2.lerp(currentCenter, desiredPosition, t);
    var newSize = vec2.lerp(currentSize, desiredSize, sizeT);

    // Apply next position
    var halfSize = newSize.uniformScale(0.5);
    follower.anchors.left = newCenter.x - halfSize.x;
    follower.anchors.right = newCenter.x + halfSize.x;
    follower.anchors.top = newCenter.y + halfSize.y;
    follower.anchors.bottom = newCenter.y - halfSize.y;

    // Apply rotation
    var smoothedRotation = quat.slerp(follower.rotation, desiredRotation, t);
    follower.rotation = smoothedRotation;
}

function setAlpha(color, alpha) {
    return new vec4(color.r, color.g, color.b, alpha);
}


function sendObjectEvent(obj, status) {
    var childCount = obj.getChildrenCount();
    for (var i = 0; i < childCount; i++) {
        var childObject = obj.getChild(i);
        sendObjectEvent(childObject, status);
    }

    var scriptComponents = obj.getComponents("Component.ScriptComponent");
    for (var j = 0; j < scriptComponents.length; j++) {
        var component = scriptComponents[j];

        if (component) {
            if (status) {
                if (component.onObjectFound) {
                    component.onObjectFound();
                }
            } else {
                if (component.onObjectLost) {
                    component.onObjectLost();
                }
            }
        }
    }
}

function wrapFunction(origFunc, newFunc) {
    if (!origFunc) {
        return newFunc;
    }
    return function() {
        origFunc();
        newFunc();
    };
}

var turnOnEvent = script.createEvent("TurnOnEvent");
turnOnEvent.bind(onLensTurnOn);

var updateEvent = script.createEvent("UpdateEvent");
updateEvent.bind(onUpdate);
