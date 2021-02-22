import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("masked_lm_prediction")
class MaskedLmPredictionLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        hiddens, _ = model(
            **sample["net_input"], features_only=True, return_all_hiddens=False
        )

        # compute MLM loss
        masked_tokens = sample["target"].ne(self.padding_idx)
        lm_outs = model.decoder.output_layer(hiddens, masked_tokens=masked_tokens)
        lm_targets = model.get_targets(sample, [lm_outs])
        lm_targets = lm_targets[masked_tokens]
        loss = F.nll_loss(
            F.log_softmax(
                lm_outs.view(-1, lm_outs.size(-1)), dim=-1, dtype=torch.float32
            ),
            lm_targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        sample_size = masked_tokens.int().sum().item()

        # compute prediction loss
        if "target_label" in sample:
            pred_outs = model.classification_heads["sentence_classification_head"](
                hiddens
            )
            label_targets = sample["target_label"].view(-1)

            pred_loss = F.nll_loss(
                F.log_softmax(pred_outs, dim=-1, dtype=torch.float32),
                label_targets,
                reduction="sum",
            )
            loss += pred_loss
            sample_size += label_targets.numel()  # count cls as another masked token

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        if "target_label" in sample:
            preds = pred_outs.max(dim=1)[1]
            logging_output.update(ncorrect=(preds == label_targets).sum().item())

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss / sample_size / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nsentences)

        return agg_output
