// Turns `:::name[optional label]` container directives into styled callout boxes.
//
//   :::def[Definition]      ->  <div class="box def"><p class="tag">Definition</p>...</div>
//   :::why                  ->  <div class="why">...</div>   (muted side note)
//
// Unknown `:name` text/leaf directives (e.g. "ratio 3:x") are turned back into
// plain text so remark-directive doesn't silently swallow them.
import { visit } from 'unist-util-visit';

const BOXES = new Set(['box', 'def', 'ok', 'warn']);

export default function remarkCallouts() {
	return (tree) => {
		visit(tree, (node, index, parent) => {
			if (node.type === 'containerDirective' && (BOXES.has(node.name) || node.name === 'why')) {
				const data = (node.data ||= {});
				data.hName = 'div';
				data.hProperties = {
					class: node.name === 'why' ? 'why' : node.name === 'box' ? 'box' : `box ${node.name}`,
				};
				const first = node.children[0];
				if (first?.data?.directiveLabel) {
					first.data.hName = 'p';
					first.data.hProperties = { class: 'tag' };
				}
				return;
			}
			if ((node.type === 'textDirective' || node.type === 'leafDirective') && parent && index !== undefined) {
				const text = { type: 'text', value: `${node.type === 'leafDirective' ? '::' : ':'}${node.name}` };
				parent.children.splice(index, 1, text, ...(node.children || []));
				return index + 1;
			}
		});
	};
}
