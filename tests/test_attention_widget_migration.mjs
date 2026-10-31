import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

let extension;
const source = readFileSync(new URL("../web/attention.js", import.meta.url), "utf8")
  .replace(/^import .*;\n/, "");
vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
class Node { configure(data) { return data; } }
extension.beforeRegisterNodeDef(Node, { name: "TuringUtilsAttentionStrategy" });
for (const [policy, refs] of [["auto", [false, true, false]], ["manual", [128]], ["none", []]]) {
  const old = ["sol", 1.3, policy, ...refs, "1x64", 1, 0, 2, 0, true, "5/5"];
  const next = new Node().configure({ widgets_values: old }).widgets_values;
  const index = 6 + refs.length;
  assert.equal(next[index], 0);
  assert.deepEqual(Array.from(next.slice(0, index)), old.slice(0, index));
  assert.deepEqual(Array.from(next.slice(index + 1)), old.slice(index));
  assert.equal(new Node().configure({ widgets_values: next }).widgets_values.length, next.length);
  const legacy = new Node().configure({ widgets_values: [...old.slice(0, -1), "off"] }).widgets_values;
  assert.equal(legacy[index], 0);
  assert.equal(legacy.at(-1), "5/5");
}
const sla = ["sla", 0.85, "none"];
assert.deepEqual(new Node().configure({ widgets_values: sla }).widgets_values, sla);
console.log("Attention widget migration: passed");
