// Turns `:::name[optional label]` container directives into styled callout boxes.
//
//   :::blue[Definition]     ->  <div class="box blue"><p class="tag">Definition</p>...</div>
//   :::muted                ->  <div class="muted">...</div>   (muted side note, no box)
//   :::qa[Question?]        ->  <details class="qa"><summary>Question?</summary>...</details>
//                               (collapsible side Q&A, closed by default)
//   :::assume[Label]        ->  <div class="assume"><p class="assume-tag">Label</p>...</div>
//                               (quiet note for an assumption or modelling choice; label defaults to "Assumption")
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
			if (node.type === 'containerDirective' && node.name === 'assume') {
				const data = (node.data ||= {});
				data.hName = 'div';
				data.hProperties = { class: 'assume' };
				const first = node.children[0];
				if (first?.data?.directiveLabel) {
					first.data.hName = 'p';
					first.data.hProperties = { class: 'assume-tag' };
				} else {
					node.children.unshift({
						type: 'paragraph',
						children: [{ type: 'text', value: 'Assumption' }],
						data: { hProperties: { class: 'assume-tag' } },
					});
				}
				return;
			}
			if (node.type === 'containerDirective' && node.name === 'qa') {
				const data = (node.data ||= {});
				data.hName = 'details';
				data.hProperties = { class: 'qa' };
				const first = node.children[0];
				if (first?.data?.directiveLabel) {
					first.data.hName = 'summary';
					// Wrap the question text so it can be styled apart from the badge and chevron.
					first.children = [
						{ type: 'qaQuestion', children: first.children, data: { hName: 'span', hProperties: { class: 'qa-q' } } },
					];
				}
				const body = node.children.filter((c) => !c.data?.directiveLabel);
				node.children = [
					...(first?.data?.directiveLabel ? [first] : []),
					{ type: 'qaBody', children: body, data: { hName: 'div', hProperties: { class: 'qa-body' } } },
				];
				return;
			}
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
