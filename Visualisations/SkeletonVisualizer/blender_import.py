import bpy
import json

json_path = r"F:\Git\Bachelor2.0\Visualisations\SkeletonVisualizer\v_e.json"
jdata = json.load(open(json_path))

verts = jdata["verts"]
edges = jdata["edges"]

me = bpy.data.meshes.new("skeleton")
me.from_pydata(verts, edges, [])
me.validate()
me.update()

ob = bpy.data.objects.new("", me)
collection = bpy.context.collection
collection.objects.link(ob)

print("ran")