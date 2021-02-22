import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("wic")
class WiCCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--classification-head-name",
            default="sentence_classification_head",
            help="name of the ranking head to use",
        )

    def forward(self, model, sample, reduce=True):

        hiddens, _ = model(
            **sample["net_input"], features_only=True, return_all_hiddens=False
        )

        embeddings = []

        # first token [CLS]
        embeddings.append(hiddens[:, 0, :])

        # other tokens
        # net_input src_ranges range1/range2
        # shape of [batch, range_len] padded with 0
        for i in range(2):
            # [batch, range_len, hidden]
            index = (
                sample["net_input"]["src_ranges"][f"range{i+1}"]
                .unsqueeze(-1)
                .expand([-1, -1, hiddens.size(-1)])
            )
            # [batch, range_len, hidden]
            mask = index != 0
            # [batch, range_len, hidden]
            embedding = hiddens.gather(dim=1, index=index) * mask
            # [batch, hidden]
            embedding = embedding.sum(dim=1) / mask.sum(dim=1)
            embeddings.append(embedding)

        concat = torch.cat(embeddings, dim=1)

        # RobertaClassificationHead expects [batch, len, hidden]
        logits = model.classification_heads["sentence_classification_head"](
            concat.unsqueeze(1)
        )
        targets = sample["target_labels"]
        sample_size = targets.numel()

        loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1), reduction="sum")

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        _, preds = logits.max(dim=1)
        logging_output.update(ncorrect=(preds == targets).sum().item())

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss_sum / sample_size / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nsentences)

        return agg_output
