// @input Component.RenderMeshVisual handMesh

script.setTriggered = function() {
    script.handMesh.mainMaterial.mainPass.baseColor = new vec4(1, 0, 0, 1);
}

script.setDefault = function() {
    script.handMesh.mainMaterial.mainPass.baseColor = new vec4(0, 1, 0, 1);
}