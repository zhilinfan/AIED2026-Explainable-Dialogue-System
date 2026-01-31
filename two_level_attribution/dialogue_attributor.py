"""
Two-Level Attribution Framework for LLM-based Dialogue Systems

Level 1: Turn-Level Attribution (Dialogue turn importance)
Level 2: Sentence-Level Attribution (Sentence importance within turns)
"""

import torch
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class DialogueTurn:
    """Represents a single dialogue turn"""
    turn_id: int
    user: str
    assistant: str

    def to_text(self) -> str:
        """Convert to text format"""
        return f"User: {self.user}\nAssistant: {self.assistant}"

    def get_user_sentences(self) -> List[str]:
        """Get list of sentences from user utterance"""
        return self._split_sentences(self.user)

    def get_assistant_sentences(self) -> List[str]:
        """Get list of sentences from assistant utterance"""
        return self._split_sentences(self.assistant)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences (supports Chinese and English)"""
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in ['。', '！', '？', '?', '；', '.', '!', '\n']:
                sent = ''.join(current).strip()
                if sent:
                    sentences.append(sent)
                current = []
        if current:
            sent = ''.join(current).strip()
            if sent:
                sentences.append(sent)
        return sentences if sentences else [text]


@dataclass
class AttributionTarget:
    """Attribution target"""
    name: str           # Target name, e.g., "behavior_type"
    text: str           # Target text
    target_type: str    # Type: "behavior", "psychology", "strategy", "custom"
    related_psychology: Optional[str] = None  # Related psychological need (for strategies)


@dataclass
class AttributionChain:
    """
    Attribution chain - represents the complete attribution path from strategy to dialogue history

    Strategy -> Psychological Need -> Dialogue History
    """
    strategy: str
    psychological_need: str
    psychology_influential_turn: int
    psychology_key_sentences: List[str]
    direct_influential_turn: int
    direct_key_sentences: List[str]


class DialogueAttributor:
    """
    Educational Dialogue Attributor

    Implements a two-level attribution framework:
    - Level 1: Turn-Level (dialogue turn importance)
    - Level 2: Sentence-Level (sentence importance within turns)
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize the dialogue attributor.

        Args:
            model: Pre-trained language model
            tokenizer: Corresponding tokenizer
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def parse_dialogue(self, dialogue_data: List[Dict]) -> List[DialogueTurn]:
        """Parse dialogue data into DialogueTurn list"""
        turns = []
        for i, turn in enumerate(dialogue_data):
            turns.append(DialogueTurn(
                turn_id=i,
                user=turn['user'],
                assistant=turn['assistant']
            ))
        return turns

    def build_dialogue_context(self, turns: List[DialogueTurn],
                                up_to_turn: Optional[int] = None,
                                exclude_turn: Optional[int] = None,
                                exclude_sentence: Optional[Tuple[int, str]] = None) -> str:
        """
        Build dialogue context.

        Args:
            turns: List of dialogue turns
            up_to_turn: Only include up to this turn
            exclude_turn: Exclude a specific turn
            exclude_sentence: Exclude a specific sentence (turn_id, sentence_text)
        """
        if up_to_turn is not None:
            turns = turns[:up_to_turn + 1]

        messages = []
        for turn in turns:
            if exclude_turn is not None and turn.turn_id == exclude_turn:
                continue

            user_text = turn.user
            if exclude_sentence is not None and turn.turn_id == exclude_sentence[0]:
                user_text = user_text.replace(exclude_sentence[1], '').strip()

            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": turn.assistant})

        # Use tokenizer's chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            context = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback: simple concatenation
            context = ""
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"

        return context

    def compute_log_likelihood(self, context: str, target_text: str) -> float:
        """
        Compute the log likelihood of target text given context.
        log p(target | context)

        Args:
            context: The dialogue context
            target_text: The target text to compute likelihood for

        Returns:
            Log likelihood value
        """
        full_text = context + target_text

        inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
        context_ids = self.tokenizer(context, return_tensors='pt').input_ids
        context_len = context_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        if context_len >= logits.shape[1]:
            return -float('inf')

        target_logits = logits[0, context_len-1:-1, :]
        target_ids = inputs.input_ids[0, context_len:]

        if target_logits.shape[0] == 0 or target_ids.shape[0] == 0:
            return -float('inf')

        log_probs = torch.log_softmax(target_logits, dim=-1)
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        return token_log_probs.sum().item()

    # ==================== Level 1: Turn-Level Attribution ====================

    def turn_level_attribution(self, turns: List[DialogueTurn],
                                target: AttributionTarget,
                                top_k: int = 1) -> Dict:
        """
        Level 1: Turn-level attribution using Temporal Likelihood Dynamics.

        Computes the marginal gain of each turn for the target output:
        g_i = psi_i - psi_{i-1}
        where psi_i = log p(target | turns[:i])

        Args:
            turns: List of dialogue turns
            target: Attribution target
            top_k: Number of top influential turns to select (default: 1)

        Returns:
            Dictionary containing turn scores and gains
        """
        results = {
            'target': target.name,
            'target_text': target.text,
            'turn_scores': [],
            'turn_gains': []
        }

        prev_score = None

        for i in range(len(turns)):
            context = self.build_dialogue_context(turns, up_to_turn=i)
            score = self.compute_log_likelihood(context, target.text)

            results['turn_scores'].append({
                'turn_id': i,
                'turn_user': turns[i].user[:50] + '...' if len(turns[i].user) > 50 else turns[i].user,
                'score': score
            })

            if prev_score is not None:
                gain = score - prev_score
                results['turn_gains'].append({
                    'turn_id': i,
                    'gain': gain
                })

            prev_score = score

        # Find the top-k turns with maximum gain
        if results['turn_gains']:
            sorted_gains = sorted(results['turn_gains'], key=lambda x: x['gain'], reverse=True)
            top_k_turns = sorted_gains[:top_k]

            # For backward compatibility, keep most_influential_turn as top-1
            results['most_influential_turn'] = top_k_turns[0]['turn_id']
            results['max_gain'] = top_k_turns[0]['gain']

            # Add top-k turns list
            results['top_k_turns'] = [t['turn_id'] for t in top_k_turns]
            results['top_k_gains'] = top_k_turns

        return results

    # ==================== Level 2: Sentence-Level Attribution ====================

    def sentence_level_attribution_loo(self, turns: List[DialogueTurn],
                                        target_turn_id: int,
                                        target: AttributionTarget) -> Dict:
        """
        Level 2: Sentence-level attribution using Leave-One-Out (LOO) method.

        Computes likelihood change when removing each sentence:
        Drop(s) = log p(target | full) - log p(target | full \ s)

        Args:
            turns: List of dialogue turns
            target_turn_id: The turn ID to analyze
            target: Attribution target

        Returns:
            Dictionary containing sentence attribution scores
        """
        target_turn = turns[target_turn_id]
        sentences = target_turn.get_user_sentences()

        full_context = self.build_dialogue_context(turns)
        baseline_score = self.compute_log_likelihood(full_context, target.text)

        results = {
            'method': 'LOO',
            'target_turn_id': target_turn_id,
            'baseline_score': baseline_score,
            'sentence_scores': []
        }

        for i, sentence in enumerate(sentences):
            modified_context = self.build_dialogue_context(
                turns,
                exclude_sentence=(target_turn_id, sentence)
            )
            modified_score = self.compute_log_likelihood(modified_context, target.text)
            drop_score = baseline_score - modified_score

            results['sentence_scores'].append({
                'sentence_id': i,
                'sentence': sentence,
                'drop_score': drop_score,
                'score': drop_score
            })

        results['sentence_scores'].sort(key=lambda x: x['drop_score'], reverse=True)
        return results

    def sentence_level_attribution_gradient(self, turns: List[DialogueTurn],
                                             target_turn_id: int,
                                             target: AttributionTarget) -> Dict:
        """
        Level 2: Sentence-level attribution using Gradient Norm method.

        Computes the gradient norm of each sentence with respect to the target output.

        Args:
            turns: List of dialogue turns
            target_turn_id: The turn ID to analyze
            target: Attribution target

        Returns:
            Dictionary containing sentence attribution scores
        """
        target_turn = turns[target_turn_id]
        sentences = target_turn.get_user_sentences()

        full_context = self.build_dialogue_context(turns)
        full_text = full_context + target.text

        inputs = self.tokenizer(full_text, return_tensors='pt').to(self.device)
        context_ids = self.tokenizer(full_context, return_tensors='pt').input_ids
        context_len = context_ids.shape[1]

        # Get embeddings and enable gradients
        embeddings = self.model.get_input_embeddings()(inputs.input_ids)
        embeddings.requires_grad_(True)

        outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs.attention_mask)
        logits = outputs.logits

        target_logits = logits[0, context_len-1:-1, :]
        target_ids = inputs.input_ids[0, context_len:]

        loss = torch.nn.functional.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_ids.reshape(-1)
        )

        loss.backward()
        gradients = embeddings.grad[0]

        results = {
            'method': 'GradientNorm',
            'target_turn_id': target_turn_id,
            'sentence_scores': []
        }

        for i, sentence in enumerate(sentences):
            sent_ids = self.tokenizer(sentence, return_tensors='pt', add_special_tokens=False).input_ids[0]
            sent_len = len(sent_ids)

            if i * sent_len < gradients.shape[0]:
                start_idx = min(i * sent_len, gradients.shape[0] - 1)
                end_idx = min(start_idx + sent_len, gradients.shape[0])
                grad_norm = gradients[start_idx:end_idx].norm().item()
            else:
                grad_norm = 0.0

            results['sentence_scores'].append({
                'sentence_id': i,
                'sentence': sentence,
                'gradient_norm': grad_norm,
                'score': grad_norm
            })

        results['sentence_scores'].sort(key=lambda x: x['gradient_norm'], reverse=True)
        return results

    # ==================== Main Attribution Methods ====================

    def attribute_single_target(self, dialogue_data: List[Dict],
                                 target_text: str,
                                 target_type: str = 'custom',
                                 method: str = 'loo',
                                 top_k: int = 1) -> Dict:
        """
        Perform attribution analysis on a single specified target.

        Args:
            dialogue_data: Dialogue data as list of dicts with 'user' and 'assistant' keys
            target_text: The target text to attribute
            target_type: Target type ('behavior', 'psychology', 'strategy', 'custom')
            method: Sentence-level attribution method ('loo', 'gradnorm', 'sim')
            top_k: Number of top influential turns to analyze (default: 1)

        Returns:
            Attribution results dictionary
        """
        turns = self.parse_dialogue(dialogue_data)

        target = AttributionTarget(
            name=f"custom_{target_type}",
            text=target_text,
            target_type=target_type
        )

        result = {
            'target_name': target.name,
            'target_text': target.text,
            'target_type': target.target_type,
            'dialogue_turns': len(turns),
            'top_k': top_k,
        }

        # Level 1: Turn-Level Attribution
        turn_attribution = self.turn_level_attribution(turns, target, top_k=top_k)
        result['level1_turn_attribution'] = turn_attribution

        # Level 2: Sentence-Level Attribution for top-k turns
        if 'top_k_turns' in turn_attribution:
            sentence_attributions = []

            for turn_id in turn_attribution['top_k_turns']:
                if method == 'loo':
                    sentence_attribution = self.sentence_level_attribution_loo(
                        turns, turn_id, target
                    )
                elif method == 'gradnorm' or method == 'gradient':
                    sentence_attribution = self.sentence_level_attribution_gradient(
                        turns, turn_id, target
                    )
                else:
                    sentence_attribution = self.sentence_level_attribution_loo(
                        turns, turn_id, target
                    )

                sentence_attributions.append(sentence_attribution)

            # For backward compatibility
            result['level2_sentence_attribution'] = sentence_attributions[0] if sentence_attributions else None
            # Add all top-k turn attributions
            result['level2_sentence_attributions'] = sentence_attributions

        return result

    def full_attribution(self, dialogue_data: List[Dict],
                         targets: List[AttributionTarget],
                         method: str = 'loo',
                         top_k: int = 1) -> Dict:
        """
        Perform full two-level attribution on multiple targets.

        Args:
            dialogue_data: Dialogue data
            targets: List of attribution targets
            method: Sentence-level attribution method
            top_k: Number of top influential turns to analyze (default: 1)

        Returns:
            Complete attribution results for all targets
        """
        turns = self.parse_dialogue(dialogue_data)

        all_results = {
            'dialogue_turns': len(turns),
            'top_k': top_k,
            'targets': []
        }

        for target in targets:
            target_result = {
                'target_name': target.name,
                'target_text': target.text,
                'target_type': target.target_type,
            }

            # Level 1
            turn_attribution = self.turn_level_attribution(turns, target, top_k=top_k)
            target_result['level1_turn_attribution'] = turn_attribution

            # Level 2 for top-k turns
            if 'top_k_turns' in turn_attribution:
                sentence_attributions = []

                for turn_id in turn_attribution['top_k_turns']:
                    if method == 'loo':
                        sentence_attribution = self.sentence_level_attribution_loo(
                            turns, turn_id, target
                        )
                    elif method == 'gradnorm' or method == 'gradient':
                        sentence_attribution = self.sentence_level_attribution_gradient(
                            turns, turn_id, target
                        )
                    else:
                        sentence_attribution = self.sentence_level_attribution_loo(
                            turns, turn_id, target
                        )

                    sentence_attributions.append(sentence_attribution)

                # For backward compatibility
                target_result['level2_sentence_attribution'] = sentence_attributions[0] if sentence_attributions else None
                target_result['level2_sentence_attributions'] = sentence_attributions

            all_results['targets'].append(target_result)

        return all_results

    def print_single_result(self, result: Dict):
        """Print attribution result for a single target"""
        print("=" * 80)
        print("Single Target Attribution Result")
        print("=" * 80)
        print(f"Dialogue turns: {result.get('dialogue_turns', 'N/A')}")
        print(f"Top-K setting: {result.get('top_k', 1)}")
        print()

        print(f"[Target]: {result['target_name']}")
        target_text = result['target_text']
        print(f"  Text: {target_text[:60]}..." if len(target_text) > 60 else f"  Text: {target_text}")
        print(f"  Type: {result['target_type']}")
        print()

        # Level 1
        if 'level1_turn_attribution' in result:
            turn_attr = result['level1_turn_attribution']
            print("[Level 1: Turn-Level Attribution]")
            print("Turn marginal gains (Top 5):")
            sorted_gains = sorted(turn_attr.get('turn_gains', []), key=lambda x: x['gain'], reverse=True)[:5]
            for gain in sorted_gains:
                print(f"  Turn {gain['turn_id']}: {gain['gain']:.4f}")

            if 'top_k_turns' in turn_attr:
                print(f"\n* Selected top-{len(turn_attr['top_k_turns'])} turns: {turn_attr['top_k_turns']}")
                for turn_id in turn_attr['top_k_turns']:
                    for ts in turn_attr.get('turn_scores', []):
                        if ts['turn_id'] == turn_id:
                            print(f"  Turn {turn_id}: {ts['turn_user']}")
            print()

        # Level 2 for all top-k turns
        if 'level2_sentence_attributions' in result:
            for sent_attr in result['level2_sentence_attributions']:
                print(f"[Level 2: Sentence-Level Attribution for Turn {sent_attr['target_turn_id']} ({sent_attr['method']})]")
                print("Sentence attribution scores:")
                for score in sent_attr['sentence_scores']:
                    sent = score['sentence']
                    sent_preview = sent[:50] + '...' if len(sent) > 50 else sent
                    if 'combined_score' in score:
                        print(f"  [{score['sentence_id']}] Combined: {score['combined_score']:.4f}")
                        print(f"      Drop: {score['drop_score']:.4f}, Hold: {score['hold_score']:.4f}")
                        print(f"      Sentence: {sent_preview}")
                    elif 'drop_score' in score:
                        print(f"  [{score['sentence_id']}] Drop: {score['drop_score']:.4f}")
                        print(f"      Sentence: {sent_preview}")
                    elif 'gradient_norm' in score:
                        print(f"  [{score['sentence_id']}] GradNorm: {score['gradient_norm']:.4f}")
                        print(f"      Sentence: {sent_preview}")
                print()

    def print_results(self, results: Dict):
        """Print attribution results for multiple targets"""
        print("=" * 80)
        print("Attribution Results")
        print("=" * 80)
        print(f"Dialogue turns: {results['dialogue_turns']}")
        print(f"Top-K setting: {results.get('top_k', 1)}")
        print()

        for target_result in results['targets']:
            print("-" * 60)
            print(f"[Target]: {target_result['target_name']}")
            text = target_result['target_text']
            print(f"  Text: {text[:60]}..." if len(text) > 60 else f"  Text: {text}")
            print(f"  Type: {target_result['target_type']}")
            print()

            turn_attr = target_result['level1_turn_attribution']
            print("  [Level 1: Turn-Level Attribution]")
            print("  Turn marginal gains (Top 3):")
            sorted_gains = sorted(turn_attr.get('turn_gains', []), key=lambda x: x['gain'], reverse=True)[:3]
            for gain in sorted_gains:
                print(f"    Turn {gain['turn_id']}: {gain['gain']:.4f}")

            if 'top_k_turns' in turn_attr:
                print(f"\n  * Selected top-{len(turn_attr['top_k_turns'])} turns: {turn_attr['top_k_turns']}")
            print()

            if 'level2_sentence_attributions' in target_result:
                for sent_attr in target_result['level2_sentence_attributions']:
                    print(f"  [Level 2: Sentence-Level for Turn {sent_attr['target_turn_id']} ({sent_attr['method']})]")
                    print("  Sentence scores (Top 3):")
                    for score in sent_attr['sentence_scores'][:3]:
                        sent_preview = score['sentence'][:35] + '...' if len(score['sentence']) > 35 else score['sentence']
                        if 'combined_score' in score:
                            print(f"    [{score['sentence_id']}] Combined: {score['combined_score']:.4f}")
                            print(f"        Drop: {score['drop_score']:.4f}, Hold: {score['hold_score']:.4f}")
                            print(f"        Sentence: {sent_preview}")
                        elif 'drop_score' in score:
                            print(f"    [{score['sentence_id']}] Drop: {score['drop_score']:.4f} | {sent_preview}")
                        elif 'gradient_norm' in score:
                            print(f"    [{score['sentence_id']}] GradNorm: {score['gradient_norm']:.4f} | {sent_preview}")
                    print()

            print()
