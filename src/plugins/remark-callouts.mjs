// Turns `:::name[optional label]` container directives into styled callout boxes.
//
//   :::blue[Definition]     ->  <div class="box blue"><p class="tag">Definition</p>...</div>
//   :::muted                ->  <div class="muted">...</div>   (muted side note, no box)
//
// Unknown `:name` text/leaf directives (e.g. "ratio 3:x") are turned back into
// plain text so remark-directive doesn't silently swallow them.
import { visit } from 'unist-util-visit';

// Directive name -> CSS class (styles in src/styles/global.css).
const CALLOUTS = {
	blue: 'box blue',
	green: 'box green',
	amber: 'box amber',
	neutral: 'box',
	muted: 'muted',
};

export default function remarkCallouts() {
	return (tree) => {
		visit(tree, (node, index, parent) => {
			if (node.type === 'containerDirective' && Object.hasOwn(CALLOUTS, node.name)) {
				const data = (node.data ||= {});
				data.hName = 'div';
				data.hProperties = { class: CALLOUTS[node.name] };
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
