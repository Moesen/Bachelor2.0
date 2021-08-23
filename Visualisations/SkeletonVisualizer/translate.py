import bpy
import bmesh

# Create empty mesh
mesh = bpy.data.meshes.new("Basic_Sphere")
basic_sphere = bpy.data.objects.new("Basic_Sphere", mesh)

# Add objects to scene
bpy.context.collection.objects.link(basic_sphere)

# Select newly created object
bpy.context.view_layer.objects.active = basic_sphere
basic_sphere.select_set(True)

# Construct bmesh and assing it 
bm = bmesh.new()
bmesh.ops.create_uvsphere(
    bm, 
    u_segments=32, 
    v_segments=16, 
    diameter=1
)
bmesh.ops.translate(
    bm,
    verts=bm.verts,
    vec = (5 ,5 , 0) 
)

bm.to_mesh(mesh)

bm.free()

print("ran")